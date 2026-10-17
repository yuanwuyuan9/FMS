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


class TimeEncoder(nn.Module):
    """Continuous cosine features, following RED-GNN Temporal's Time_Embed_1."""
    def __init__(self, dim):
        super().__init__()
        self.frequency = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim)).float())
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, times):
        return torch.cos(times[..., None] * self.frequency + self.phase)


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
        # Saved configurations without this field retain their original scoring.
        self.feature_normalization = getattr(args, 'feature_normalization', 'none')
        if self.feature_normalization not in ('none', 'layer_norm'):
            raise ValueError('Unknown classifier feature normalization.')
        self.sampling_seed = args.seed
        self.sampling_time_scale = getattr(args, 'sampling_time_scale', 30.0)
        self.history_time_scale = getattr(args, 'history_time_scale', 30.0)
        if any(not math.isfinite(value) or value <= 0 for value in
               (self.sampling_time_scale, self.history_time_scale)):
            raise ValueError('Temporal decay scales must be finite and positive.')
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
        self.use_flow = not getattr(args, 'no_flow', False)
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
        self.cfm = matchers[self.cfm_variant](sigma=sigma) if self.use_flow else None

        self.set_graph(params_for_neighbors)

    def _build_model(self):
        self._build_relation_feature()
        self.time_encoder = TimeEncoder(self.hidden_dim)
        self.time_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())

        if self.use_flow:
            self.MLP = MLP(dim=3 * self.hidden_dim, time_varying=True,
                           w=self.cfm_model_dim, out_dim=self.hidden_dim)
        classifier_dim = (3 if self.use_flow else 2) * self.hidden_dim
        self.Linear_out_head = nn.Linear(classifier_dim, self.n_entity)
        self.history_gate_weight = nn.Parameter(torch.zeros(classifier_dim))
        self.history_gate_bias = nn.Parameter(torch.zeros(()))

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
        # Graphs stay indexed on CPU; checkpoints contain learned parameters only.
        self.graph = params_for_neighbors

    @property
    def relation_features(self):
        # Rebuild on each use: caching this tensor would retain a stale autograd graph.
        zeros = self.relation_features_base.new_zeros(1, self.hidden_dim)
        return torch.cat((self.relation_features_base, zeros), dim=0)

    def _aggregate_neighbors(self, edge_vectors, masks):
        batch_size = len(edge_vectors[0])
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

    def encode_context(self, entities, relations, timestamps):
        ids, deltas, masks = self.graph.sample_tree(
            entities.detach().cpu().numpy(), timestamps.detach().cpu().numpy(),
            self.context_hops, self.neighbor_samples, self.sampling_seed,
            self.sampling_time_scale)
        device = entities.device
        vectors = [self.relation_features_base[relations][:, None]]
        tensor_masks = []
        for relation_ids, delta, mask in zip(ids, deltas, masks):
            valid = torch.as_tensor(mask, device=device)
            relation_ids = torch.as_tensor(relation_ids, device=device)
            delta = torch.as_tensor(delta, device=device)
            temporal = self.time_projection(self.time_encoder(delta))
            features = self.relation_features[relation_ids] + temporal
            vectors.append(features.masked_fill(~valid[..., None], 0.0))
            tensor_masks.append(valid)
        return self._aggregate_neighbors(vectors, tensor_masks)

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

    def _predict(self, heads, relations, timestamps, return_endpoint=False):
        head_context = self.encode_context(heads, relations, timestamps)
        relation_emb = self.relation_features_base[relations]
        features = [head_context, relation_emb]
        endpoint = None
        if self.use_flow:
            endpoint = self._flow_endpoint(head_context, relation_emb)
            features.append(endpoint)
        features = self._classification_features(*features)
        scores = self.Linear_out_head(features)
        scores = self._fuse_history(scores, features, heads, relations, timestamps)
        if return_endpoint:
            return scores, head_context, endpoint
        return scores, head_context

    def _classification_features(self, *features):
        # The classifier and history gate share normalized inputs, not normalized flow targets.
        if self.feature_normalization == 'layer_norm':
            features = tuple(F.layer_norm(feature, (self.hidden_dim,), eps=1e-5) for feature in features)
        return torch.cat(features, dim=-1)

    def _fuse_history(self, logits, features, heads, relations, timestamps):
        history = self.graph.historical_tail_distribution(
            heads.detach().cpu().numpy(), relations.detach().cpu().numpy(),
            timestamps.detach().cpu().numpy(), self.history_time_scale)
        history = torch.as_tensor(history, device=logits.device, dtype=logits.dtype)
        log_generation = F.log_softmax(logits, dim=-1)
        gate_logits = (features * self.history_gate_weight).sum(dim=-1, keepdim=True) + self.history_gate_bias
        # A missing history must fall back exactly to the generation branch.
        available = history.sum(dim=-1, keepdim=True) > 0
        log_history = history.clamp_min(torch.finfo(history.dtype).tiny).log()
        log_history = log_history.masked_fill(history == 0, -torch.inf)
        fused = torch.logaddexp(F.logsigmoid(-gate_logits) + log_generation,
                               F.logsigmoid(gate_logits) + log_history)
        return torch.where(available, fused, log_generation)

    def forward(self, heads, relations, timestamps):
        """Return [batch, n_entities] scores without accepting answer labels."""
        return self._predict(heads, relations, timestamps)[0]

    def flow_matching_loss(self, head_context, relations, tails, timestamps, tail_context=None):
        # Answers are supervision only; this branch never feeds the classifier.
        if tail_context is None:
            with torch.no_grad():
                tail_context = self.encode_context(tails, relations, timestamps)
        source = head_context.detach()
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(source, tail_context)
        vt_pred = self._velocity(xt, t, head_context, self.relation_features_base[relations])
        return F.mse_loss(vt_pred, ut)

    @staticmethod
    def train_step(model, optimizer, batch, return_components=False):
        model.train()
        optimizer.zero_grad()
        timestamps = batch['timestamps']
        scores, head_context, endpoint = model._predict(
            batch['heads'], batch['relations'], timestamps, return_endpoint=True)
        # Exclude other training positives from the CE denominator, retaining this target.
        classification_scores = scores.clone()
        if 'other_answers' in batch:
            classification_scores[batch['other_answers']] = -torch.inf
        classification_loss = F.cross_entropy(classification_scores, batch['labels'])
        loss = classification_loss
        flow_loss = scores.new_zeros(())
        endpoint_loss = scores.new_zeros(())
        tail_context = None
        if model.use_flow and (model.lamda != 0 or model.endpoint_weight != 0):
            with torch.no_grad():
                tail_context = model.encode_context(batch['labels'], batch['relations'], timestamps)
        if model.use_flow and model.lamda != 0:
            flow_loss = model.flow_matching_loss(
                head_context, batch['relations'], batch['labels'], timestamps, tail_context)
            loss = loss + model.lamda * flow_loss
        if model.use_flow and model.endpoint_weight != 0:
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
            scores = model(batch['heads'], batch['relations'], batch['timestamps'])
            correct = batch['labels'] == scores.argmax(dim=1)
        return correct.cpu(), scores
