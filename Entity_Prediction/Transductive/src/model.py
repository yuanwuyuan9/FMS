import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggregators import MeanAggregator, ConcatAggregator, CrossAggregator, AttentionAggregator
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.models import MLP


class FMS(nn.Module):
    def __init__(self, args, n_relations, params_for_neighbors, n_entity):
        super(FMS, self).__init__()
        self._parse_args(args, n_relations, params_for_neighbors, n_entity)
        self._build_model()

    def _parse_args(self, args, n_relations, params_for_neighbors, n_entity):
        if not args.use_context:
            raise ValueError("Entity prediction requires --use_context.")
        if args.context_hops < 1 or args.neighbor_samples < 1 or args.flow_steps < 1:
            raise ValueError("context_hops, neighbor_samples and flow_steps must be positive.")
        if args.feature_type != 'id':
            raise ValueError(f"Unsupported feature_type: {args.feature_type}")

        self.n_relations = n_relations
        self.n_entity = n_entity
        self.hidden_dim = args.dim
        self.tok_k = args.tok_k
        self.num_heads = args.num_heads
        self.neighbor_agg = args.neighbor_agg
        self.cfm_model_dim = args.cfm_model_dim
        self.neighbor_samples = args.neighbor_samples
        self.context_hops = args.context_hops
        self.flow_steps = args.flow_steps
        # Tail prediction uses outgoing context and semantic neighbor weights.
        if getattr(args, 'semantic_weighting', 'softmax') != 'softmax':
            raise ValueError('Legacy mean weighting is no longer supported.')
        if getattr(args, 'context_direction', 'outgoing') != 'outgoing':
            raise ValueError('Only outgoing context expansion is supported.')
        self.semantic_weighting = 'softmax'
        self.context_direction = 'outgoing'
        self.lamda = args.lamda
        self.classification_loss = 'ce'
        # Older checkpoints do not contain an endpoint loss weight.
        self.endpoint_weight = getattr(args, 'endpoint_weight', 0.0)
        if not math.isfinite(self.endpoint_weight) or self.endpoint_weight < 0:
            raise ValueError('endpoint_weight must be finite and nonnegative.')
        if getattr(args, 'no_flow', False):
            raise ValueError('This checkpoint does not contain the FMS flow predictor.')
        self.cfm_variant = args.cfm_variant
        matchers = {
            'CFM': ConditionalFlowMatcher,
            'TargetCFM': TargetConditionalFlowMatcher,
            'VPCFM': VariancePreservingConditionalFlowMatcher,
        }
        if self.cfm_variant not in matchers:
            raise ValueError(
                "Use CFM, TargetCFM or VPCFM. Global OTCFM/SBCFM coupling changes "
                "the supervised (head, relation, tail) pairs and is not supported."
            )
        sigma = 0.0 if self.cfm_variant == 'VPCFM' else args.cfm_sigma
        if sigma < 0 or (self.cfm_variant == 'TargetCFM' and sigma >= 1):
            raise ValueError("sigma must be nonnegative and below 1 for TargetCFM.")
        self.cfm = matchers[self.cfm_variant](sigma=sigma)

        self.set_graph(params_for_neighbors)

    def _build_model(self):
        self._build_relation_feature()
        self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())

        self.MLP = MLP(dim=3 * self.hidden_dim, time_varying=True,
                       w=self.cfm_model_dim, out_dim=self.hidden_dim)
        self.Linear_out_head = nn.Linear(3 * self.hidden_dim, self.n_entity)

    def _build_relation_feature(self):
        self.relation_features_base = nn.Parameter(torch.empty(self.n_relations, self.hidden_dim))
        nn.init.xavier_uniform_(self.relation_features_base)

    def _get_neighbor_aggregators(self):
        if self.neighbor_agg == 'mean':
            aggregator_class = MeanAggregator
        elif self.neighbor_agg == 'concat':
            aggregator_class = ConcatAggregator
        elif self.neighbor_agg == 'cross':
            aggregator_class = CrossAggregator
        elif self.neighbor_agg == 'attention':
            aggregator_class = AttentionAggregator
        else:
            raise ValueError('Unknown neighbor aggregator: %s' % self.neighbor_agg)

        aggregators = []
        for hop in range(self.context_hops):
            params = {'input_dim': self.hidden_dim, 'output_dim': self.hidden_dim, 'k': self.tok_k}
            if hop == self.context_hops - 1:
                params['self_included'] = False
            else:
                params['act'] = F.relu
            if self.neighbor_agg == 'attention':
                params['num_heads'] = self.num_heads
            aggregators.append(aggregator_class(**params))
        return aggregators

    def set_graph(self, params_for_neighbors):
        """Replace sampled graph buffers without replacing learned parameters."""
        device = next(self.parameters(), torch.empty(0)).device
        triples = np.column_stack((params_for_neighbors[1], params_for_neighbors[2]))
        _, edge_groups = np.unique(triples, axis=0, return_inverse=True)
        for name, values in zip(('entity2edges', 'edge2entities', 'edge2relation', 'edge_groups'),
                                list(params_for_neighbors) + [edge_groups]):
            tensor = torch.as_tensor(values, dtype=torch.long, device=device)
            if name in self._buffers:
                setattr(self, name, tensor)
            else:
                self.register_buffer(name, tensor)
        self.null_edge = len(self.edge2relation) - 1

    @property
    def relation_features(self):
        # Rebuild on each use: caching this tensor would retain a stale autograd graph.
        zeros = self.relation_features_base.new_zeros(1, self.hidden_dim)
        return torch.cat((self.relation_features_base, zeros), dim=0)

    def _get_neighbors_and_masks(self, relations, entities, train_edges=None):
        batch_size = len(entities)
        edge_list = [relations.reshape(batch_size, 1)]
        masks = []
        if train_edges is not None:
            if train_edges.shape != entities.shape:
                raise ValueError("Each training query must supply one target edge ID.")
            if torch.any((train_edges < 0) | (train_edges >= self.null_edge)):
                raise ValueError("Invalid training target edge ID.")
            blocked_groups = self.edge_groups[train_edges].unsqueeze(1)

        for hop in range(self.context_hops):
            if hop == 0:
                neighbor_entities = entities
            else:
                neighbor_entities = self.edge2entities[edge_list[-1], 1]
            neighbor_edges = self.entity2edges[neighbor_entities].reshape(batch_size, -1)
            valid = neighbor_edges != self.null_edge
            if train_edges is not None:
                valid = valid & (self.edge_groups[neighbor_edges] != blocked_groups)
            # Mask before expanding the next hop. Targets and their inverses
            # are already excluded from the training graph by the data loader.
            neighbor_edges = neighbor_edges.masked_fill(~valid, self.null_edge)
            edge_list.append(neighbor_edges)
            masks.append(valid)
        return edge_list, masks

    def _aggregate_neighbors(self, edge_list, masks):
        batch_size = len(edge_list[0])
        features = self.relation_features
        edge_vectors = [features[edge_list[0]]]
        edge_vectors.extend(features[self.edge2relation[edges]] for edges in edge_list[1:])
        for i, aggregator in enumerate(self.aggregators):
            next_vectors = []
            for hop in range(self.context_hops - i):
                parent_count = edge_vectors[hop].shape[1]
                # Each parent has S outgoing children from its target endpoint.
                neighbors = edge_vectors[hop + 1].reshape(
                    batch_size, parent_count, -1, self.hidden_dim)
                neighbor_masks = masks[hop].reshape(batch_size, parent_count, -1, 1)
                next_vectors.append(aggregator(edge_vectors[hop], neighbors, neighbor_masks))
            edge_vectors = next_vectors
        return edge_vectors[0].reshape(batch_size, self.hidden_dim)

    def encode_context(self, entities, relations, train_edges=None):
        edges, masks = self._get_neighbors_and_masks(relations, entities, train_edges)
        return self._aggregate_neighbors(edges, masks)

    def _velocity(self, state, time, head_context, relation_emb):
        return self.MLP(torch.cat((state, head_context, relation_emb, time[:, None]), dim=-1))

    def _flow_endpoint(self, head_context, relation_emb):
        # Deterministic mean-start Euler rollout, shared by training and inference.
        state = (torch.zeros_like(head_context) if self.cfm_variant == 'TargetCFM'
                 else head_context)
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            time = state.new_full((len(state),), step * dt)
            state = state + dt * self._velocity(state, time, head_context, relation_emb)
        return state

    def _predict(self, heads, relations, train_edges=None, return_endpoint=False):
        head_context = self.encode_context(heads, relations, train_edges)
        relation_emb = self.relation_features_base[relations]
        endpoint = self._flow_endpoint(head_context, relation_emb)
        scores = self.Linear_out_head(torch.cat((head_context, relation_emb, endpoint), dim=-1))
        if return_endpoint:
            return scores, head_context, endpoint
        return scores, head_context

    def forward(self, heads, relations, train_edges=None):
        """Return [batch, n_entities] scores without accepting answer labels."""
        return self._predict(heads, relations, train_edges)[0]

    def flow_matching_loss(self, head_context, relations, tails, train_edges, tail_context=None):
        # Answers are supervision only; this branch never feeds the classifier.
        if tail_context is None:
            with torch.no_grad():
                tail_context = self.encode_context(tails, relations, train_edges)
        source = head_context.detach()
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(source, tail_context)
        vt_pred = self._velocity(xt, t, head_context, self.relation_features_base[relations])
        return F.mse_loss(vt_pred, ut)

    @staticmethod
    def train_step(model, optimizer, batch, return_components=False):
        model.train()
        optimizer.zero_grad()
        train_edges = batch.get('train_edges')
        scores, head_context, endpoint = model._predict(
            batch['heads'], batch['relations'], train_edges, return_endpoint=True)
        classification_loss = F.cross_entropy(scores, batch['labels'])
        loss = classification_loss
        flow_loss = scores.new_zeros(())
        endpoint_loss = scores.new_zeros(())
        tail_context = None
        if model.lamda != 0 or model.endpoint_weight != 0:
            with torch.no_grad():
                tail_context = model.encode_context(batch['labels'], batch['relations'], train_edges)
        if model.lamda != 0:
            flow_loss = model.flow_matching_loss(
                head_context, batch['relations'], batch['labels'], train_edges, tail_context)
            loss = loss + model.lamda * flow_loss
        if model.endpoint_weight != 0:
            endpoint_loss = F.mse_loss(endpoint, tail_context)
            loss = loss + model.endpoint_weight * endpoint_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite FMS training loss.")
        loss.backward()
        optimizer.step()
        if return_components:
            return dict(total=loss.item(), classification=classification_loss.item(),
                        cfm=flow_loss.item(), endpoint=endpoint_loss.item(),
                        weighted_cfm=model.lamda * flow_loss.item(),
                        weighted_endpoint=model.endpoint_weight * endpoint_loss.item())
        return loss.item()

    @staticmethod
    def test_step(model, batch):
        model.eval()
        with torch.no_grad():
            scores = model(batch['heads'], batch['relations'])
            correct = batch['labels'] == scores.argmax(dim=1)
        return correct.cpu(), scores
