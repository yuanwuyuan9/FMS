import torch
import torch.nn as nn
from abc import abstractmethod
import torch.nn.functional as F
import math
import numpy as np
import warnings


class Aggregator(nn.Module):
    """
    Base class for different aggregation strategies in a GNN.
    Includes Top-K neighbor selection based on semantic importance,
    inspired by DiffusionE's energy module.
    """

    def __init__(self, input_dim, output_dim, act, self_included, k,
                 semantic_g_dim=32,  # Dimension for the g_layer output
                 semantic_tau=0.95):  # Temperature for semantic scoring
        super(Aggregator, self).__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if k is None:
            raise ValueError("k (number of neighbors) must be specified (e.g., int > 0 or 0 for all).")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included
        self.k = int(k)

        if semantic_tau <= 0:
            raise ValueError("semantic_tau must be positive.")
        self.semantic_tau = semantic_tau
        self.semantic_g_dim = semantic_g_dim or input_dim
        self.g_layer = nn.Linear(self.input_dim, self.semantic_g_dim, bias=False)
        nn.init.xavier_uniform_(self.g_layer.weight)

    def _calculate_semantic_scores(self, self_vectors_flat, neighbor_vectors_flat):
        """Return negative-distance logits; normalize only after masking/selection."""
        g_self = self.g_layer(self_vectors_flat).unsqueeze(1)
        g_neighbors = self.g_layer(neighbor_vectors_flat)
        squared_l2_diff = (g_neighbors - g_self).square().sum(dim=-1)
        return -squared_l2_diff / self.semantic_tau

    def forward(self, self_vectors, neighbor_vectors, masks):
        """
        Select Top-K neighbors, then aggregate with differentiable semantic weights.
        Hard indices are discrete; selected logits carry gradients to g_layer.
        """
        if (neighbor_vectors.ndim < 3 or masks.ndim != neighbor_vectors.ndim
                or self_vectors.shape[-1] != self.input_dim
                or neighbor_vectors.shape[-1] != self.input_dim
                or masks.shape[-1] != 1
                or self_vectors.shape[:-1] != neighbor_vectors.shape[:-2]
                or masks.shape[:-1] != neighbor_vectors.shape[:-1]):
            raise ValueError('Expected self [..., D], neighbors [..., N, D], masks [..., N, 1].')
        original_self_shape = self_vectors.shape
        entity_dims = self_vectors.shape[:-1]
        num_entities = math.prod(entity_dims)
        n_neighbor_max = neighbor_vectors.shape[-2]
        self_vectors_flat = self_vectors.reshape(num_entities, self.input_dim)
        neighbor_vectors_flat = neighbor_vectors.reshape(num_entities, n_neighbor_max, self.input_dim)
        masks_flat = masks.reshape(num_entities, n_neighbor_max, 1)

        # --- Neighbor Aggregation ---
        if n_neighbor_max == 0 or num_entities == 0:
            entity_vectors_aggregated = torch.zeros(num_entities, self.input_dim, device=self_vectors.device,
                                                    dtype=self_vectors.dtype)
        else:
            use_top_k_selection = self.k > 0 and self.k < n_neighbor_max
            scores = self._calculate_semantic_scores(self_vectors_flat, neighbor_vectors_flat)
            scores = scores.masked_fill(~masks_flat.squeeze(-1).bool(), -float('inf'))

            selected_neighbors, selected_masks = neighbor_vectors_flat, masks_flat
            selected_scores = scores
            if use_top_k_selection:
                _, top_indices = torch.topk(scores, k=self.k, dim=-1)
                selected_neighbors = torch.gather(
                    neighbor_vectors_flat, 1, top_indices.unsqueeze(-1).expand(-1, -1, self.input_dim))
                selected_masks = torch.gather(masks_flat, 1, top_indices.unsqueeze(-1))
                selected_scores = torch.gather(scores, 1, top_indices)

            valid = selected_masks.squeeze(-1).bool()
            # Avoid softmax(-inf, ..., -inf): empty rows contribute exactly zero.
            safe_scores = selected_scores.masked_fill(~valid.any(dim=-1, keepdim=True), 0.0)
            weights = torch.softmax(safe_scores, dim=-1).masked_fill(~valid, 0.0)
            entity_vectors_aggregated = (selected_neighbors * weights.unsqueeze(-1)).sum(dim=1)

        target_agg_shape = tuple(entity_dims) + (1, self.input_dim)
        entity_vectors_reshaped = entity_vectors_aggregated.reshape(target_agg_shape)
        self_vectors_reshaped = self_vectors_flat.reshape(original_self_shape)

        outputs = self._call(self_vectors_reshaped, entity_vectors_reshaped)
        return self.act(outputs)

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        pass

class ConcatAggregator(Aggregator):
    """
    Aggregator that concatenates the self vector (if enabled) with the
    semantically weighted neighbor vector, followed by a linear layer.
    """
    def __init__(self, input_dim, output_dim, k, act=lambda x: x, self_included=True):
        super(ConcatAggregator, self).__init__(input_dim, output_dim, act, self_included, k)

        neighbor_agg_dim = self.input_dim
        total_input_dim = neighbor_agg_dim + self.input_dim if self.self_included else neighbor_agg_dim

        if total_input_dim <= 0:
             warnings.warn("ConcatAggregator initialized with zero total input dimension.")
             self.layer = None
        else:
            self.layer = nn.Linear(total_input_dim, self.output_dim)
            nn.init.xavier_uniform_(self.layer.weight)
            if self.layer.bias is not None:
                nn.init.zeros_(self.layer.bias)

    def _call(self, self_vectors, entity_vectors):
        entity_vectors_squeezed = entity_vectors.squeeze(-2) # Shape: [*, input_dim]
        original_shape = self_vectors.shape # Or entity_vectors_squeezed.shape

        if self.self_included:
            combined_features = torch.cat([self_vectors, entity_vectors_squeezed], dim=-1)
        else:
            combined_features = entity_vectors_squeezed

        if self.layer is not None and combined_features.shape[-1] > 0 :
            if combined_features.shape[-1] != self.layer.in_features:
                 raise RuntimeError(f"ConcatAgg _call: Dim mismatch! Feature dim {combined_features.shape[-1]} != Layer input dim {self.layer.in_features}")

            # Reshape for Linear layer
            N_total = np.prod(original_shape[:-1]) if len(original_shape) > 1 else 1
            output_from_layer = self.layer(combined_features.reshape(N_total, -1))
            # Reshape back
            output = output_from_layer.reshape(original_shape[:-1] + (self.output_dim,))
        else:
            # Handle zero-dimensional input or missing layer
            target_shape = original_shape[:-1] + (self.output_dim,)
            output = torch.zeros(target_shape, device=combined_features.device, dtype=combined_features.dtype)
            if self.layer is None and combined_features.shape[-1] > 0:
                 warnings.warn("ConcatAggregator._call: Layer not initialized properly, returning zeros.")

        return output


class MeanAggregator(Aggregator):
    """
    Aggregator that adds the self vector (if enabled) to the semantically weighted
    neighbor vector, followed by a linear layer.
    """
    def __init__(self, input_dim, output_dim, k, act=lambda x: x, self_included=True):
        super(MeanAggregator, self).__init__(input_dim, output_dim, act, self_included, k)
        self.layer = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        if self.layer.bias is not None:
             nn.init.zeros_(self.layer.bias)

    def _call(self, self_vectors, entity_vectors):
        entity_vectors_squeezed = entity_vectors.squeeze(-2) # Shape: [*, input_dim]
        original_shape = self_vectors.shape

        if self.self_included:
            combined = self_vectors + entity_vectors_squeezed
        else:
            combined = entity_vectors_squeezed

        # Reshape for Linear layer
        N_total = np.prod(original_shape[:-1]) if len(original_shape) > 1 else 1
        output_from_layer = self.layer(combined.reshape(N_total, self.input_dim))

        # Reshape back
        output = output_from_layer.reshape(original_shape[:-1] + (self.output_dim,))

        return output


class CrossAggregator(Aggregator):
    """
    Aggregator that performs self-interaction (outer product) on the
    semantically weighted neighbor vector, potentially concatenates the self_vector,
    and applies a linear layer.
    """
    def __init__(self, input_dim, output_dim, k, act=lambda x: x, self_included=True):
        super(CrossAggregator, self).__init__(input_dim, output_dim, act, self_included, k)

        interaction_dim = self.input_dim * self.input_dim
        self_dim = self.input_dim if self.self_included else 0
        total_input_dim = interaction_dim + self_dim

        if total_input_dim <= 0:
             warnings.warn("CrossAggregator initialized with zero total input dimension.")
             self.layer = None
        else:
             self.layer = nn.Linear(total_input_dim, self.output_dim)
             nn.init.xavier_uniform_(self.layer.weight)
             if self.layer.bias is not None:
                 nn.init.zeros_(self.layer.bias)

    def _call(self, self_vectors, entity_vectors):
        entity_vectors_squeezed = entity_vectors.squeeze(-2) # Shape: [*, input_dim]
        original_shape = self_vectors.shape
        N_total = np.prod(original_shape[:-1]) if len(original_shape) > 1 else 1

        # Reshape for interaction calculation
        agg_neighbors_flat = entity_vectors_squeezed.reshape(N_total, self.input_dim)

        # Calculate self-interaction
        interaction = torch.bmm(agg_neighbors_flat.unsqueeze(2), agg_neighbors_flat.unsqueeze(1))
        interaction_flat = interaction.view(N_total, -1) # Shape: [N, D*D]

        if self.self_included:
            self_vectors_flat = self_vectors.reshape(N_total, self.input_dim)
            combined_features = torch.cat([self_vectors_flat, interaction_flat], dim=-1)
        else:
            combined_features = interaction_flat

        if self.layer is not None and combined_features.shape[-1] > 0:
             if combined_features.shape[-1] != self.layer.in_features:
                  raise RuntimeError(f"CrossAgg _call: Dim mismatch! Feature dim {combined_features.shape[-1]} != Layer input dim {self.layer.in_features}")
             output_from_layer = self.layer(combined_features)
             output = output_from_layer.reshape(original_shape[:-1] + (self.output_dim,))
        else:
             target_shape = original_shape[:-1] + (self.output_dim,)
             output = torch.zeros(target_shape, device=combined_features.device, dtype=combined_features.dtype)
             if self.layer is None and combined_features.shape[-1] > 0:
                  warnings.warn("CrossAggregator._call: Layer not initialized properly, returning zeros.")

        return output


class AttentionAggregator(Aggregator):
    """
    Attention Aggregator (Adapted).

    Note: This is NOT standard GAT. It operates on the pre-aggregated
    neighbor vector provided by the base class `forward` method.

    It calculates multi-head attention scores based on the interaction
    between the self_vector and the single aggregated entity_vector.
    These scores act as gates/modulators for the entity_vector before
    it's potentially combined with the self_vector.

    """

    def __init__(self, input_dim, output_dim, k,
                 act=lambda x: F.leaky_relu(x, 0.2),
                 self_included=True,
                 num_heads=4,
                 dropout_rate=0.1,
                 leaky_relu_slope=0.2):
        super(AttentionAggregator, self).__init__(input_dim, output_dim, act, self_included, k)

        self.num_heads = num_heads
        self.leaky_relu_slope = leaky_relu_slope

        # Calculate actual head dimension, ensuring it can cover output_dim
        self.head_dim_actual = math.ceil(output_dim / num_heads)
        # self.internal_mh_dim is the dimension after W_self/W_entity and before final projection
        self.internal_mh_dim = self.num_heads * self.head_dim_actual
        # Transformation applied to self_vectors and entity_vectors
        # Output dim allows splitting into heads
        self.W_self = nn.Linear(input_dim, self.internal_mh_dim, bias=False)
        self.W_entity = nn.Linear(input_dim, self.internal_mh_dim, bias=False)
        # Attention scoring layer: Takes concatenated transformed self & entity features per head
        self.attention_layer = nn.Linear(2 * self.head_dim_actual, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(self.leaky_relu_slope)
        self.dropout = nn.Dropout(dropout_rate)  # Applied to gating scores
        self.output_projection = nn.Linear(self.internal_mh_dim, output_dim, bias=True)

        # Initialization
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_entity.weight)
        nn.init.xavier_uniform_(self.attention_layer.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def _call(self, self_vectors, entity_vectors):
        """
        Implements the adapted attention mechanism on pre-aggregated neighbors.

        Args:
            self_vectors: Tensor shape [*, input_dim]
            entity_vectors: Tensor shape [*, 1, input_dim] (semantically weighted neighbors)

        Returns:
            outputs: Tensor shape [*, output_dim]
        """
        entity_vectors_squeezed = entity_vectors.squeeze(-2)  # Shape: [*, input_dim]
        original_shape_prefix = self_vectors.shape[:-1]
        N_total = np.prod(original_shape_prefix) if len(original_shape_prefix) > 0 else 1

        # 1. Flatten and Transform
        self_vectors_flat = self_vectors.reshape(N_total, self.input_dim)
        entity_vectors_flat = entity_vectors_squeezed.reshape(N_total, self.input_dim)

        # Transformed features will have dimension self.internal_mh_dim
        h_self_flat = self.W_self(self_vectors_flat)  # [N_total, internal_mh_dim]
        h_entity_flat = self.W_entity(entity_vectors_flat)  # [N_total, internal_mh_dim]

        # 2. Reshape for Multi-Head
        # [N_total, num_heads, head_dim_actual]
        h_self_multihead = h_self_flat.view(N_total, self.num_heads, self.head_dim_actual)
        h_entity_multihead = h_entity_flat.view(N_total, self.num_heads, self.head_dim_actual)

        # 3. Calculate Attention/Gating Scores
        # Concat features per head: [N_total, num_heads, 2 * head_dim_actual]
        attn_input = torch.cat([h_self_multihead, h_entity_multihead], dim=-1)
        # Compute logits: [N_total, num_heads, 1]
        attn_logits = self.leaky_relu(self.attention_layer(attn_input))

        # 4. Apply Sigmoid for Gating (instead of Softmax)
        gate_scores = torch.sigmoid(attn_logits)  # Shape: [N_total, num_heads, 1]
        gate_scores = self.dropout(gate_scores)

        # 5. Apply Gating/Modulation
        # Modulate the transformed entity features
        # Shape: [N_total, num_heads, head_dim_actual]
        weighted_entity_multihead = gate_scores * h_entity_multihead

        # 6. Combine Heads
        # Concatenate heads: [N_total, internal_mh_dim]
        weighted_entity_flat_internal = weighted_entity_multihead.contiguous().view(N_total, self.internal_mh_dim)

        # 7. Combine with Self (Optional)
        if self.self_included:
            # Add the *transformed* self vector (which is also internal_mh_dim)
            intermediate_output_flat = h_self_flat + weighted_entity_flat_internal
        else:
            intermediate_output_flat = weighted_entity_flat_internal

        # 8. Final Projection
        # Project from internal_mh_dim to the final output_dim
        output_flat = self.output_projection(intermediate_output_flat)  # [N_total, output_dim]

        # 9. Reshape to Original Leading Dims
        output_final_shape = original_shape_prefix + (self.output_dim,)
        output = output_flat.reshape(output_final_shape)

        return output