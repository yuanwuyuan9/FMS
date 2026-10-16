import torch
import torch.nn as nn
import torch.nn.functional as F

from aggregators import MeanAggregator, ConcatAggregator, CrossAggregator, AttentionAggregator
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.models import MLP


class QueryPropagationLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(QueryPropagationLayer, self).__init__()
        self.Ws_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wr_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wq_attn = nn.Linear(hidden_dim, hidden_dim)
        self.w_alpha = nn.Linear(hidden_dim, 1)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        hidden,
        edge_heads,
        edge_tails,
        edge_relations,
        edge_ids,
        query_relations,
        mask_edges,
        relation_features,
    ):
        source_hidden = hidden.index_select(1, edge_heads)
        edge_relation_features = torch.index_select(relation_features, 0, edge_relations)
        query_relation_features = torch.index_select(relation_features, 0, query_relations)

        attention_input = (
            self.Ws_attn(source_hidden)
            + self.Wr_attn(edge_relation_features).unsqueeze(0)
            + self.Wq_attn(query_relation_features).unsqueeze(1)
        )
        attention = torch.sigmoid(self.w_alpha(F.relu(attention_input)))
        messages = (source_hidden + edge_relation_features.unsqueeze(0)) * attention

        edge_mask = ~(edge_ids.unsqueeze(0).unsqueeze(-1) == mask_edges.unsqueeze(1)).any(dim=-1)
        messages = messages * edge_mask.to(messages.dtype).unsqueeze(-1)

        aggregated = hidden.new_zeros(hidden.shape)
        aggregated.index_add_(1, edge_tails, messages)
        return F.relu(hidden + self.W_h(aggregated))


class FMS(nn.Module):
    def __init__(self, args, n_relations, params_for_neighbors, n_entity):
        super(FMS, self).__init__()
        self._parse_args(args, n_relations, params_for_neighbors, n_entity)
        self._build_model()
        self.MLP = MLP(
            dim=self.hidden_dim,
            time_varying=True,
            w=self.cfm_model_dim,
            out_dim=self.hidden_dim,
        ).to(self.device)
        self.query_seed_layer = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.node_init_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.score_layer = nn.Linear(self.hidden_dim, 1, bias=False)

    def _parse_args(self, args, n_relations, params_for_neighbors, n_entity):
        self.n_relations = n_relations
        self.n_entity = n_entity
        self.use_gpu = args.cuda
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.feature_type = args.feature_type
        self.tok_k = args.tok_k
        self.num_heads = args.num_heads
        self.lamda = args.lamda

        self.neighbor_samples = args.neighbor_samples
        self.context_hops = args.context_hops
        if args.neighbor_agg == "mean":
            self.neighbor_agg = MeanAggregator
        elif args.neighbor_agg == "concat":
            self.neighbor_agg = ConcatAggregator
        elif args.neighbor_agg == "cross":
            self.neighbor_agg = CrossAggregator
        elif args.neighbor_agg == "attention":
            self.neighbor_agg = AttentionAggregator
        else:
            raise ValueError(f"Unknown neighbor_agg: {args.neighbor_agg}")

        self.cfm_model_dim = args.cfm_model_dim
        self.cfm_variant = args.cfm_variant
        self.cfm_sigma = args.cfm_sigma
        self.cfm_ot_method = args.cfm_ot_method
        self.inference_t_strategy = args.inference_t_strategy
        self.inference_t = args.inference_t
        self.inference_mc_samples = max(1, args.inference_mc_samples)

        sigma = self.cfm_sigma
        if self.cfm_variant == "CFM":
            self.cfm = ConditionalFlowMatcher(sigma=sigma)
        elif self.cfm_variant == "OTCFM":
            self.cfm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        elif self.cfm_variant == "TargetCFM":
            self.cfm = TargetConditionalFlowMatcher(sigma=sigma)
        elif self.cfm_variant == "SBCFM":
            self.cfm = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method=self.cfm_ot_method)
        elif self.cfm_variant == "VPCFM":
            self.cfm = VariancePreservingConditionalFlowMatcher(sigma=0.0)
        else:
            raise ValueError(f"Unknown CFM variant: {self.cfm_variant}")

        self.set_context_graph(params_for_neighbors, n_entity)
        print(f"Training with {self.cfm_variant} (sigma={sigma if self.cfm_variant != 'VPCFM' else 'N/A'})")

    def set_context_graph(self, params_for_neighbors, n_entity=None):
        if params_for_neighbors is None:
            raise ValueError("Context graph parameters are required.")
        self.entity2edges = torch.as_tensor(params_for_neighbors[0], dtype=torch.long, device=self.device)
        self.edge2entities = torch.as_tensor(params_for_neighbors[1], dtype=torch.long, device=self.device)
        self.edge2relation = torch.as_tensor(params_for_neighbors[2], dtype=torch.long, device=self.device)
        if n_entity is not None:
            self.n_entity = n_entity
        self._set_propagation_edges()

    def _set_propagation_edges(self):
        edge_heads = self.edge2entities[:, 0]
        edge_tails = self.edge2entities[:, 1]
        valid_edges = (
            (edge_heads < self.n_entity)
            & (edge_tails < self.n_entity)
            & (self.edge2relation < self.n_relations)
        )
        valid_indices = torch.nonzero(valid_edges, as_tuple=False).view(-1)

        self.propagation_edge_ids = valid_indices
        self.propagation_edge_heads = edge_heads.index_select(0, valid_indices)
        self.propagation_edge_tails = edge_tails.index_select(0, valid_indices)
        self.propagation_edge_relations = self.edge2relation.index_select(0, valid_indices)

    def _build_model(self):
        self._build_relation_feature()
        self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())
        self.propagation_layers = nn.ModuleList(
            [QueryPropagationLayer(self.hidden_dim) for _ in range(self.context_hops)]
        )

    def forward(self, batch):
        self.known_entities = batch["known_entities"].to(self.device)
        self.train_edges = batch["train_edges"].to(self.device)
        self.relations = batch["relations"].to(self.device)
        target_entities = batch.get("target_entities")
        self.target_entities = target_entities.to(self.device) if target_entities is not None else None
        self._call_model()

    def _call_model(self):
        relation_features = self._relation_features()

        # All queries share entity contexts, so exclude every target edge in the
        # minibatch from both context encoding and query propagation.
        masked_edges = self.train_edges.reshape(-1)
        masked_edges = torch.unique(masked_edges[masked_edges >= 0])
        if masked_edges.numel() > 0:
            self.train_edges = masked_edges.unsqueeze(0).expand(self.known_entities.shape[0], -1)

        edge_list_known, mask_list_known = self._get_neighbors_and_masks(
            self.relations,
            self.known_entities,
            self.train_edges,
        )
        known_context = self._aggregate_neighbors_with_features(
            edge_list_known,
            mask_list_known,
            relation_features,
        )
        relation_context = relation_features[self.relations]
        query_flow = self._query_flow(known_context, relation_context)
        query_seed = self.query_seed_layer(torch.cat([known_context, relation_context, query_flow], dim=-1))

        entity_contexts = self._all_entity_contexts(relation_features)
        entity_states = self._propagate_query(query_seed, entity_contexts, relation_features)
        self.scores = self.score_layer(entity_states).squeeze(-1)

        if self.training and self.target_entities is not None:
            target_contexts = entity_contexts[
                torch.arange(entity_contexts.shape[0], device=entity_contexts.device),
                self.target_entities,
            ]
            self.loss_cfm = self._target_cfm_loss(known_context, target_contexts)
        else:
            self.loss_cfm = torch.zeros((), device=known_context.device, dtype=known_context.dtype)

    def _query_flow(self, known_context, relation_context):
        if self.training:
            t = torch.rand(known_context.shape[0], device=known_context.device, dtype=known_context.dtype)
            xt = self.cfm.compute_mu_t(known_context, relation_context, t)
            return self.MLP(torch.cat([xt, t[:, None]], dim=-1))
        return self._infer_flow(known_context, relation_context)

    def _target_cfm_loss(self, known_context, target_contexts):
        t, xt, ut, _, _, _ = self.cfm.guided_sample_location_and_conditional_flow(
            known_context,
            target_contexts,
            known_context,
            target_contexts,
            return_noise=True,
        )
        t = t.to(known_context.device)
        xt = xt.to(known_context.device)
        ut = ut.to(known_context.device)
        vt_pred = self.MLP(torch.cat([xt, t[:, None]], dim=-1))
        return torch.mean((vt_pred - ut) ** 2)

    def _propagate_query(self, query_seed, entity_contexts, relation_features):
        hidden = F.relu(self.node_init_layer(entity_contexts))
        seed_hidden = hidden.new_zeros(hidden.shape)
        batch_ids = torch.arange(hidden.shape[0], device=hidden.device)
        seed_hidden[batch_ids, self.known_entities] = query_seed
        hidden = hidden + seed_hidden

        mask_edges = self.train_edges
        if mask_edges.dim() == 1:
            mask_edges = mask_edges.unsqueeze(-1)

        for propagation_layer in self.propagation_layers:
            hidden = propagation_layer(
                hidden,
                self.propagation_edge_heads,
                self.propagation_edge_tails,
                self.propagation_edge_relations,
                self.propagation_edge_ids,
                self.relations,
                mask_edges,
                relation_features,
            )
        return hidden

    def _all_entity_contexts(self, relation_features):
        query_batch_size = self.known_entities.shape[0]
        entities = torch.arange(self.n_entity, device=self.device, dtype=torch.long)
        relations = torch.full(
            (self.n_entity,),
            self.n_relations,
            device=self.device,
            dtype=torch.long,
        )
        # _call_model makes the mask identical for all queries in the batch.
        train_edges = self.train_edges[:1].reshape(1, -1).expand(self.n_entity, -1)

        original_batch_size = self.batch_size
        self.batch_size = self.n_entity
        try:
            edge_list, mask_list = self._get_neighbors_and_masks(relations, entities, train_edges)
            entity_contexts = self._aggregate_neighbors_with_features(edge_list, mask_list, relation_features)
        finally:
            self.batch_size = original_batch_size

        return entity_contexts.unsqueeze(0).expand(query_batch_size, -1, -1)

    def _infer_flow(self, x0, x1):
        if self.inference_t_strategy == "random":
            t, xt, _, _, _, _ = self.cfm.guided_sample_location_and_conditional_flow(
                x0,
                x1,
                x0,
                x1,
                return_noise=True,
            )
            t = t.to(x0.device)
            xt = xt.to(x0.device)
            return self.MLP(torch.cat([xt, t[:, None]], dim=-1))

        if self.inference_t_strategy == "fixed":
            t = torch.full((x0.shape[0],), self.inference_t, device=x0.device, dtype=x0.dtype)
            xt = self.cfm.compute_mu_t(x0, x1, t)
            return self.MLP(torch.cat([xt, t[:, None]], dim=-1))

        if self.inference_t_strategy == "mc":
            preds = []
            for _ in range(self.inference_mc_samples):
                t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
                xt = self.cfm.compute_mu_t(x0, x1, t)
                preds.append(self.MLP(torch.cat([xt, t[:, None]], dim=-1)))
            return torch.stack(preds, dim=0).mean(dim=0)

        raise ValueError(f"Unknown inference_t_strategy: {self.inference_t_strategy}")

    def _build_relation_feature(self):
        if self.feature_type != "id":
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")

        relation_embeddings = torch.empty(self.n_relations, self.hidden_dim, device=self.device)
        nn.init.xavier_uniform_(relation_embeddings)
        self.relation_features_base = nn.Parameter(relation_embeddings)

        zeros = torch.zeros(1, self.hidden_dim, device=self.device)
        self.register_buffer("null_relation_feature", zeros)

    def _relation_features(self):
        return torch.cat([self.relation_features_base, self.null_relation_feature], dim=0)

    def _get_neighbors_and_masks(self, relations, entities, train_edges):
        edges_list = [relations]
        masks = []
        train_edges = train_edges.to(entities.device)
        if train_edges.dim() == 1:
            train_edges = train_edges.unsqueeze(-1)

        for hop in range(self.context_hops):
            if hop == 0:
                neighbor_entities = entities
            else:
                neighbor_entities = torch.index_select(
                    self.edge2entities,
                    0,
                    edges_list[-1].reshape(-1),
                ).view([self.batch_size, -1])

            neighbor_edges = torch.index_select(
                self.entity2edges,
                0,
                neighbor_entities.reshape(-1),
            ).view([self.batch_size, -1])
            edges_list.append(neighbor_edges)

            matches = neighbor_edges.unsqueeze(-1) == train_edges.unsqueeze(1)
            mask = (~matches.any(dim=-1)).float()
            masks.append(mask)

        return edges_list, masks

    def _get_neighbor_aggregators(self):
        aggregators = []
        base_params_intermediate = {
            "input_dim": self.hidden_dim,
            "output_dim": self.hidden_dim,
            "k": self.tok_k,
            "act": F.relu,
        }
        base_params_last_hop = {
            "input_dim": self.hidden_dim,
            "output_dim": self.hidden_dim,
            "k": self.tok_k,
            "self_included": False,
        }

        if self.context_hops == 1:
            params = base_params_last_hop.copy()
            if self.neighbor_agg == AttentionAggregator:
                params["num_heads"] = self.num_heads
            aggregators.append(self.neighbor_agg(**params))
        else:
            params = base_params_intermediate.copy()
            if self.neighbor_agg == AttentionAggregator:
                params["num_heads"] = self.num_heads
            aggregators.append(self.neighbor_agg(**params))

            for _ in range(self.context_hops - 2):
                params = base_params_intermediate.copy()
                if self.neighbor_agg == AttentionAggregator:
                    params["num_heads"] = self.num_heads
                aggregators.append(self.neighbor_agg(**params))

            params = base_params_last_hop.copy()
            if self.neighbor_agg == AttentionAggregator:
                params["num_heads"] = self.num_heads
            aggregators.append(self.neighbor_agg(**params))

        return aggregators

    def _aggregate_neighbors(self, edge_list, mask_list):
        return self._aggregate_neighbors_with_features(edge_list, mask_list, self._relation_features())

    def _aggregate_neighbors_with_features(self, edge_list, mask_list, relation_features):
        index_tensor = edge_list[0].to(relation_features.device)
        edge_vectors = [torch.index_select(relation_features, 0, index_tensor)]

        for edges in edge_list[1:]:
            relations = torch.index_select(
                self.edge2relation,
                0,
                edges.reshape(-1),
            ).view(list(edges.shape) + [-1])
            edge_vectors.append(
                torch.index_select(
                    relation_features,
                    0,
                    relations.reshape(-1),
                ).view(list(relations.shape) + [-1])
            )

        for layer_idx in range(self.context_hops):
            aggregator = self.aggregators[layer_idx]
            next_edge_vectors = []
            neighbors_shape = [self.batch_size, -1, 1, self.neighbor_samples, aggregator.input_dim]
            masks_shape = [self.batch_size, -1, 1, self.neighbor_samples, 1]

            for hop in range(self.context_hops - layer_idx):
                vector = aggregator(
                    self_vectors=edge_vectors[hop],
                    neighbor_vectors=edge_vectors[hop + 1].view(neighbors_shape),
                    masks=mask_list[hop].view(masks_shape),
                )
                next_edge_vectors.append(vector)
            edge_vectors = next_edge_vectors

        return edge_vectors[0].view([self.batch_size, self.hidden_dim])
