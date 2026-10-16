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
                 use_semantic_topk=True,  # Flag to enable semantic top-k
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

        self.use_semantic_topk = use_semantic_topk
        self.semantic_tau = semantic_tau

        if self.use_semantic_topk:
            if semantic_g_dim is None:
                # Default semantic_g_dim to input_dim if not specified
                warnings.warn("semantic_g_dim not specified for semantic Top-K, defaulting to input_dim.")
                self.semantic_g_dim = input_dim
            else:
                self.semantic_g_dim = semantic_g_dim
            # Each entity/relation (self and neighbor) will be passed through this
            self.g_layer = nn.Linear(self.input_dim, self.semantic_g_dim, bias=False)
            nn.init.xavier_uniform_(self.g_layer.weight)
            # Score = exp( -||g(self) - g(neighbor)||^2 / tau )
        else:
            self.g_layer = None
            self.semantic_g_dim = None

    def _calculate_semantic_scores(self, self_vectors_flat, neighbor_vectors_flat):
        """
        Calculates semantic importance scores based on DiffusionE's energy module concept.
        Score = exp( -||g(self) - g(neighbor)||^2 / tau )
        where g is a linear transformation.
        A higher score means higher semantic importance (smaller difference in g-space).
        """
        # self_vectors_flat: [num_entities, input_dim]
        # neighbor_vectors_flat: [num_entities, n_neighbor, input_dim]

        g_self = self.g_layer(self_vectors_flat)  # [num_entities, semantic_g_dim]
        # Expand self_g for broadcasting with neighbors
        g_self_expanded = g_self.unsqueeze(1)  # [num_entities, 1, semantic_g_dim]

        # Transform neighbor vectors
        # Need to reshape neighbors to pass through g_layer
        num_entities, n_neighbor, _ = neighbor_vectors_flat.shape
        g_neighbors = self.g_layer(neighbor_vectors_flat.reshape(num_entities * n_neighbor, self.input_dim))
        g_neighbors = g_neighbors.reshape(num_entities, n_neighbor,
                                          self.semantic_g_dim)  # [num_entities, n_neighbor, semantic_g_dim]

        # Calculate squared L2 norm of the difference in g-space
        # diff_g = g_self_expanded - g_neighbors # [num_entities, n_neighbor, semantic_g_dim]
        # squared_l2_diff = torch.sum(diff_g * diff_g, dim=-1) # [num_entities, n_neighbor]

        # Alternative: using dot product for similarity after g-transform
        # This is closer to standard attention but with a learned projection g.
        # scores = torch.sum(g_self_expanded * g_neighbors, dim=-1) # [num_entities, n_neighbor]
        # return scores / math.sqrt(self.semantic_g_dim) # Optional scaling
        diff_g = g_neighbors - g_self_expanded  # [num_entities, n_neighbor, semantic_g_dim]
        squared_l2_diff = torch.sum(diff_g * diff_g, dim=-1)  # [num_entities, n_neighbor]

        # scores = -squared_l2_diff # Higher (less negative) is better
        # To match the exp structure:
        scores = torch.exp(-squared_l2_diff / self.semantic_tau)  # Higher (closer to 1) is better

        return scores  # Shape: [num_entities, n_neighbor]

    def forward(self, self_vectors, neighbor_vectors, masks):
        """
        Performs Top-K neighbor selection (if k > 0) followed by mean aggregation,
        then calls the specific aggregation logic in _call.
        Top-K can be based on dot-product (default) or semantic importance (if enabled).
        """
        original_self_shape = self_vectors.shape
        original_neighbor_shape = neighbor_vectors.shape
        original_mask_shape = masks.shape

        if not (len(original_neighbor_shape) >= 3 and len(original_mask_shape) >= 3):
            raise ValueError(
                f"neighbor_vectors (shape {original_neighbor_shape}) and masks (shape {original_mask_shape}) must have at least 3 dimensions ([..., n_neighbor, feature/1]).")
        if original_self_shape[-1] != self.input_dim:
            raise ValueError(f"self_vectors last dim ({original_self_shape[-1]}) != self.input_dim ({self.input_dim})")
        if original_neighbor_shape[-1] != self.input_dim:
            raise ValueError(
                f"neighbor_vectors last dim ({original_neighbor_shape[-1]}) != self.input_dim ({self.input_dim})")
        if original_mask_shape[-1] != 1:
            raise ValueError(f"masks last dim ({original_mask_shape[-1]}) should be 1")

        n_neighbor_input_dim = original_mask_shape[-2]
        if original_neighbor_shape[-2] != n_neighbor_input_dim:
            warnings.warn(
                f"Neighbor dimension mismatch detected: neighbor_vectors.shape[-2] ({original_neighbor_shape[-2]}) "
                f"!= masks.shape[-2] ({n_neighbor_input_dim}). Using mask's dimension {n_neighbor_input_dim}.")

        entity_dims = original_self_shape[:-1]
        num_entities = np.prod(entity_dims) if len(entity_dims) > 0 else 1

        try:
            self_vectors_flat = self_vectors.reshape(num_entities, self.input_dim)
            if neighbor_vectors.numel() == 0:
                if num_entities > 0:
                    n_neighbor_max = 0
                    neighbor_vectors_flat = torch.empty(num_entities, 0, self.input_dim, device=neighbor_vectors.device,
                                                        dtype=neighbor_vectors.dtype)
                    masks_flat = torch.empty(num_entities, 0, 1, device=masks.device, dtype=masks.dtype)
                else:
                    n_neighbor_max = 0
                    neighbor_vectors_flat = torch.empty(0, 0, self.input_dim, device=neighbor_vectors.device,
                                                        dtype=neighbor_vectors.dtype)
                    masks_flat = torch.empty(0, 0, 1, device=masks.device, dtype=masks.dtype)

            elif num_entities == 0:
                raise ValueError("num_entities is 0, but neighbor_vectors is not empty.")
            else:
                elements_per_entity_neighbor = neighbor_vectors.numel() // num_entities
                elements_per_entity_mask = masks.numel() // num_entities
                if elements_per_entity_neighbor % self.input_dim != 0:
                    raise ValueError(
                        f"Neighbor elements per entity ({elements_per_entity_neighbor}) not divisible by input_dim ({self.input_dim}).")
                calculated_n_neighbor = elements_per_entity_neighbor // self.input_dim
                if elements_per_entity_mask != calculated_n_neighbor:
                    if n_neighbor_input_dim == calculated_n_neighbor:
                        warnings.warn(f"Inconsistent neighbor count inferred. Vectors imply {calculated_n_neighbor}, "
                                      f"mask elements imply {elements_per_entity_mask}, but mask shape[-2] implies {n_neighbor_input_dim}. "
                                      f"Proceeding with {calculated_n_neighbor}.")
                    else:
                        raise ValueError(
                            f"Inconsistent neighbor count inferred and mask shape mismatch. Vectors imply {calculated_n_neighbor}, mask elements imply {elements_per_entity_mask}, mask shape[-2] is {n_neighbor_input_dim}.")
                n_neighbor_max = calculated_n_neighbor
                neighbor_vectors_flat = neighbor_vectors.reshape(num_entities, n_neighbor_max, self.input_dim)
                masks_flat = masks.reshape(num_entities, n_neighbor_max, 1)
        except (RuntimeError, ValueError) as e:
            # ... (error handling same as before)
            print("--- Aggregator Reshape Error Debug Info ---")
            print(f"Original self_vectors shape: {original_self_shape}")
            print(f"Original neighbor_vectors shape: {original_neighbor_shape}")
            print(f"Original masks shape: {original_mask_shape}")
            print(f"Calculated num_entities: {num_entities}")
            print(
                f"Input n_neighbor_max (from mask): {n_neighbor_input_dim if 'n_neighbor_input_dim' in locals() else 'Not Available'}")
            print(
                f"Inferred/Used n_neighbor_max: {n_neighbor_max if 'n_neighbor_max' in locals() else 'Not Calculated'}")
            print(f"Expected self.input_dim: {self.input_dim}")
            print(f"Actual neighbor_vectors numel: {neighbor_vectors.numel()}")
            print(f"Actual masks numel: {masks.numel()}")
            print("--- End Reshape Error Debug Info ---")
            raise e

        # --- Neighbor Aggregation ---
        if n_neighbor_max == 0 or num_entities == 0:
            entity_vectors_aggregated = torch.zeros(num_entities, self.input_dim, device=self_vectors.device,
                                                    dtype=self_vectors.dtype)
        else:
            use_top_k_selection = self.k > 0 and self.k < n_neighbor_max
            if use_top_k_selection:
                if self.use_semantic_topk and self.g_layer is not None:
                    # --- Semantic Top-K Scoring ---
                    # No need for no_grad() here as g_layer needs to be trained
                    scores = self._calculate_semantic_scores(self_vectors_flat, neighbor_vectors_flat)
                else:
                    # --- Standard Dot-Product Top-K Scoring ---
                    with torch.no_grad():  # Dot product similarity doesn't involve trainable params for scoring
                        self_vectors_expanded = self_vectors_flat.unsqueeze(1)
                        scores = torch.sum(self_vectors_expanded * neighbor_vectors_flat, dim=-1)

                # Apply mask to scores before topk
                scores = scores.masked_fill(~masks_flat.squeeze(-1).bool(),
                                            -float('inf'))  # Low score for masked neighbors

                safe_k = min(self.k, n_neighbor_max)
                try:
                    # topk returns (values, indices)
                    _, top_indices = torch.topk(scores, k=safe_k, dim=-1)
                except RuntimeError as e_topk:
                    warnings.warn(
                        f"torch.topk failed with k={safe_k} on scores of shape {scores.shape}. Error: {e_topk}. Returning zero aggregation.")
                    top_indices = torch.zeros(num_entities, 0, dtype=torch.long, device=scores.device)

                if top_indices.shape[1] > 0:
                    # Gather selected neighbors and their original masks
                    top_indices_expanded_vec = top_indices.unsqueeze(-1).expand(-1, -1, self.input_dim)
                    selected_neighbors = torch.gather(neighbor_vectors_flat, dim=1, index=top_indices_expanded_vec)

                    top_indices_expanded_mask = top_indices.unsqueeze(-1)  # For gathering from masks_flat
                    selected_masks = torch.gather(masks_flat, dim=1, index=top_indices_expanded_mask)

                    # Also gather the scores of the selected top-k items to further ensure only valid ones are used
                    gathered_scores_for_masking = torch.gather(scores, dim=1, index=top_indices)
                    # Ensure that selected_masks only count if the score was not -inf (i.e., originally unmasked and selected)
                    selected_masks = selected_masks * (gathered_scores_for_masking > -float('inf')).unsqueeze(-1)

                else:  # No neighbors selected by topk (e.g., if all scores were -inf or k=0 effectively)
                    selected_neighbors = torch.empty(num_entities, 0, self.input_dim, device=self_vectors.device,
                                                     dtype=self_vectors.dtype)
                    selected_masks = torch.empty(num_entities, 0, 1, device=self_vectors.device, dtype=masks.dtype)

                # Aggregate the selected neighbors
                if selected_neighbors.shape[1] > 0:
                    sum_selected_neighbors = torch.sum(selected_neighbors * selected_masks, dim=1)
                    num_selected_neighbors = torch.sum(selected_masks, dim=1)
                    num_selected_neighbors = torch.clamp(num_selected_neighbors, min=1e-9)  # Avoid division by zero
                    entity_vectors_aggregated = sum_selected_neighbors / num_selected_neighbors
                else:
                    entity_vectors_aggregated = torch.zeros_like(self_vectors_flat)  # No valid neighbors after top-k
            else:  # No Top-K, aggregate all valid neighbors
                sum_neighbors = torch.sum(neighbor_vectors_flat * masks_flat, dim=1)
                num_neighbors_valid = torch.sum(masks_flat, dim=1)
                num_neighbors_valid = torch.clamp(num_neighbors_valid, min=1e-9)
                entity_vectors_aggregated = sum_neighbors / num_neighbors_valid

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
    mean-aggregated neighbor vector, followed by a linear layer.
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
    Aggregator that adds the self vector (if enabled) to the mean-aggregated
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
    mean-aggregated neighbor vector, potentially concatenates the self_vector,
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
            entity_vectors: Tensor shape [*, 1, input_dim] (mean-aggregated neighbors)

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