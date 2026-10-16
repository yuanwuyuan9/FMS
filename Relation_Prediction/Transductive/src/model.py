import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregators import MeanAggregator, ConcatAggregator, CrossAggregator, AttentionAggregator

# Import your modules
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher
)
from torchcfm.models.models import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class FMS(nn.Module):
    def __init__(self, args, n_relations, params_for_neighbors, n_entity):
        super(FMS, self).__init__()
        self._parse_args(args, n_relations, params_for_neighbors, n_entity)
        self._build_model()

        self.MLP = MLP(dim=self.n_relations, time_varying=True, w=self.cfm_model_dim, out_dim=self.n_relations).to(device)
        self.Linear_out = nn.Linear(self.n_relations * 2, self.n_relations)


    def _parse_args(self, args, n_relations, params_for_neighbors, n_entity):
        self.n_relations = n_relations
        self.n_entity = n_entity
        self.use_gpu = args.cuda

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.feature_type = args.feature_type
        self.tok_k = args.tok_k
        self.num_heads = args.num_heads
        self.lamda = args.lamda

        self.use_context = args.use_context
        if self.use_context:
            self.entity2edges = torch.LongTensor(params_for_neighbors[0]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[0])
            self.edge2entities = torch.LongTensor(params_for_neighbors[1]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[1])
            self.edge2relation = torch.LongTensor(params_for_neighbors[2]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[2])
            self.neighbor_samples = args.neighbor_samples
            self.context_hops = args.context_hops
            if args.neighbor_agg == 'mean':
                self.neighbor_agg = MeanAggregator
            elif args.neighbor_agg == 'concat':
                self.neighbor_agg = ConcatAggregator
            elif args.neighbor_agg == 'cross':
                self.neighbor_agg = CrossAggregator
            elif args.neighbor_agg == 'attention':
                self.neighbor_agg = AttentionAggregator

        self.cfm_model_dim = args.cfm_model_dim
        self.cfm_variant = args.cfm_variant
        self.cfm_sigma = args.cfm_sigma
        self.cfm_ot_method = args.cfm_ot_method

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

        print(f"Training with {self.cfm_variant} (sigma={sigma if self.cfm_variant != 'VPCFM' else 'N/A'})")
    def _build_model(self):


        if self.use_context:
            self._build_relation_feature()
            self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())

    def forward(self, batch):
        if self.use_context:
            self.entity_pairs = batch['entity_pairs']
            self.train_edges = batch['train_edges']

        self.labels = batch['labels']
        self._call_model()

    def _call_model(self):
        self.scores = 0.

        if self.use_context:
            edge_list_head, mask_list_head = self._get_neighbors_and_masks(self.labels, self.entity_pairs[:, 0],
                                                                           self.train_edges)
            edge_list_tail, mask_list_tail = self._get_neighbors_and_masks(self.labels, self.entity_pairs[:, 1],
                                                                           self.train_edges)

            self.aggregated_neighbors_head = self._aggregate_neighbors(edge_list_head, mask_list_head)
            self.aggregated_neighbors_tail = self._aggregate_neighbors(edge_list_tail, mask_list_tail)

            output = torch.cat((self.aggregated_neighbors_head, self.aggregated_neighbors_tail), dim=-1)
            self.output = self.Linear_out(output)

            x0 = self.aggregated_neighbors_head
            x1 = self.aggregated_neighbors_tail

            t, xt, ut, y0, y1, _ = self.cfm.guided_sample_location_and_conditional_flow(x0, x1, x0, x1,
                                                                                        return_noise=True)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)
            y0 = y0.to(device)
            y1 = y1.to(device)

            vt_pred = self.MLP(torch.cat([xt, t[:, None]], dim=-1))

            self.loss_cfm = torch.mean((vt_pred - ut) ** 2)

            self.scores += self.output * vt_pred

    def _build_relation_feature(self):
        if self.feature_type == 'id':
            relation_embeddings = torch.empty(self.n_relations, self.hidden_dim)
            nn.init.xavier_uniform_(relation_embeddings)

            if self.use_gpu:
                relation_embeddings = relation_embeddings.cuda()
            else:
                relation_embeddings = relation_embeddings
            self.relation_features_base = nn.Parameter(relation_embeddings)
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")

        zeros = torch.zeros(1, self.hidden_dim)
        if self.use_gpu:
            zeros = zeros.cuda()

        self.relation_features = torch.cat([self.relation_features_base, zeros], dim=0)

    def _get_neighbors_and_masks(self, relations, entity_pairs, train_edges):

        edges_list = [relations]
        masks = []
        # Relation predictions need to mask associated edges
        train_edges = torch.unsqueeze(train_edges, -1)

        for i in range(self.context_hops):
            if i == 0:
                neighbor_entities = entity_pairs
            else:

                neighbor_entities = torch.index_select(self.edge2entities, 0,
                                                       edges_list[-1].view(-1)).view([self.batch_size, -1])

            neighbor_edges = torch.index_select(self.entity2edges, 0,
                                                neighbor_entities.view(-1)).view([self.batch_size, -1])
            edges_list.append(neighbor_edges)
            mask = neighbor_edges - train_edges
            mask = (mask != 0).float()
            masks.append(mask)

        return edges_list, masks

    def _get_neighbor_aggregators(self):
        aggregators = []
        # Define generic parameters for aggregators in the middle tier
        base_params_intermediate = {
            'input_dim': self.hidden_dim,
            'output_dim': self.hidden_dim,
            'k': self.tok_k,
            'act': F.relu,
        }
        # Define generic parameters for the last layer of aggregators
        base_params_last_hop = {
            'input_dim': self.hidden_dim,
            'output_dim': self.n_relations,
            'k': self.tok_k,
            'self_included': False,
        }

        if self.context_hops == 1:
            params = base_params_last_hop.copy()
            if self.neighbor_agg == AttentionAggregator:
                params['num_heads'] = self.num_heads
            aggregators.append(self.neighbor_agg(**params))
        else:
            params = base_params_intermediate.copy()
            if self.neighbor_agg == AttentionAggregator:
                params['num_heads'] = self.num_heads
            aggregators.append(self.neighbor_agg(**params))

            for i in range(self.context_hops - 2):
                params = base_params_intermediate.copy()
                if self.neighbor_agg == AttentionAggregator:
                    params['num_heads'] = self.num_heads
                aggregators.append(self.neighbor_agg(**params))

            params = base_params_last_hop.copy()
            if self.neighbor_agg == AttentionAggregator:
                params['num_heads'] = self.num_heads
            aggregators.append(self.neighbor_agg(**params))

        return aggregators

    def _aggregate_neighbors(self, edge_list, mask_list):

        device = self.relation_features.device
        index_tensor = edge_list[0].to(device)
        edge_vectors = [torch.index_select(self.relation_features, 0, index_tensor)]

        for edges in edge_list[1:]:
            relations = torch.index_select(self.edge2relation, 0, edges.view(-1)).view(list(edges.shape) + [-1])
            edge_vectors.append(torch.index_select(self.relation_features, 0, relations.view(-1)).view(list(relations.shape) + [-1]))

        for i in range(self.context_hops):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            neighbors_shape = [self.batch_size, -1, 1, self.neighbor_samples, aggregator.input_dim]
            masks_shape = [self.batch_size, -1, 1, self.neighbor_samples, 1]

            for hop in range(self.context_hops - i):

                vector = aggregator(self_vectors=edge_vectors[hop],
                                    neighbor_vectors=edge_vectors[hop + 1].view(neighbors_shape),
                                    masks=mask_list[hop].view(masks_shape))
                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter
        res = edge_vectors[0].view([self.batch_size, self.n_relations])

        return res


    @staticmethod
    def train_step(model, optimizer, batch):
        model.train()
        optimizer.zero_grad()
        model(batch)
        criterion = nn.CrossEntropyLoss()
        loss = torch.mean(criterion(model.scores, model.labels)) + model.lamda * model.loss_cfm
        loss.backward()
        optimizer.step()

        return loss.item()


    @staticmethod
    def test_step(model, batch):
        model.eval()
        with torch.no_grad():
            model(batch)
            correct_predictions = (model.labels == model.scores.argmax(dim=1))
            return correct_predictions, model.scores






