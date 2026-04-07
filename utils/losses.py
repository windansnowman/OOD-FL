
from __future__ import print_function

import torch
import torch.nn as nn

import logging

logger = logging.getLogger("logger")


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning:
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, scale_weight=1, fac_label=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)

            mask_scale = mask.clone().detach()
            mask_cross_feature = torch.ones_like(mask_scale).to(device)

            for ind, label in enumerate(labels.view(-1)):
                if label == fac_label:
                    mask_scale[ind, :] = mask[ind, :] * scale_weight

        else:
            mask = mask.float().to(device)

        contrast_feature = features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
        elif self.contrast_mode == 'all':
            anchor_feature = features
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) * mask_cross_feature
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos_mask = (mask_scale * log_prob).sum(1)
        mask_check = mask.sum(1)
        for ind, mask_item in enumerate(mask_check):
            if mask_item == 0:
                continue
            else:
                mask_check[ind] = 1 / mask_item
        mask_apply = mask_check
        mean_log_prob_pos = mean_log_prob_pos_mask * mask_apply
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss
# from __future__ import print_function
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F  # For potential normalization
# import logging
#
# logger = logging.getLogger("logger")
#
#
# class SupConLoss(nn.Module):
#     """
#     Supervised Contrastive Loss enhanced to specifically pull poisoned samples' features
#     towards the features of clean samples of their (poisoned) target class.
#     """
#
#     def __init__(self, temperature=0.07, base_temperature=0.07,
#                  pull_to_clean_strength=1, normalize_features=False):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.base_temperature = base_temperature
#         self.pull_to_clean_strength = pull_to_clean_strength
#         self.normalize_features = normalize_features  # Flag to normalize features
#
#     def forward(self, features,
#                 original_labels=None,  # Original labels BEFORE any poisoning
#                 modified_labels=None,  # Main labels for SupCon (poisoned labels for backdoor samples)
#                 target_label_for_poison=None):  # The single integer class index for the backdoor target
#         """
#         Args:
#             features: hidden vector of shape [bsz, feature_dim].
#             original_labels: torch.Tensor of shape [bsz]. Original class labels.
#             modified_labels: torch.Tensor of shape [bsz]. Labels used for SupCon;
#                              for poisoned samples, this is the target_label_for_poison.
#             target_label_for_poison: int. The specific class label that poisoned samples are targeted to.
#         """
#         device = features.device
#         batch_size = features.shape[0]
#
#         if modified_labels is None:
#             raise ValueError("`modified_labels` must be provided for SupCon learning.")
#         if self.pull_to_clean_strength > 0 and (original_labels is None or target_label_for_poison is None):
#             logger.warning("`pull_to_clean_strength` > 0 but `original_labels` or `target_label_for_poison` "
#                            "is None. Auxiliary pull loss will not be computed.")
#
#         # Optional: L2 Normalize features
#         if self.normalize_features:
#             features = F.normalize(features, p=2, dim=1)
#
#         # --- Part 0: Determine meta_labels (0 for clean, 1 for poisoned) ---
#         # A sample is considered poisoned if its original label differs from its modified label.
#         # And its modified label must be the target_label_for_poison (stricter check, optional)
#         meta_labels = torch.zeros_like(modified_labels, dtype=torch.long, device=device)
#         if original_labels is not None and modified_labels is not None:
#             # Basic definition: if original and modified differ, it's considered manipulated for poisoning.
#             meta_labels = (original_labels != modified_labels).long()
#             # Stricter check (optional): a sample is poisoned IF original != modified AND modified == target_label
#             # if target_label_for_poison is not None:
#             #     meta_labels = ((original_labels != modified_labels) & (modified_labels == target_label_for_poison)).long()
#
#         # --- Part 1: Standard Supervised Contrastive Loss (using modified_labels) ---
#         # `modified_labels` are used to define positive/negative pairs for SupCon
#         current_labels_for_supcon = modified_labels.contiguous().view(-1, 1)
#         if current_labels_for_supcon.shape[0] != batch_size:
#             raise ValueError('Num of modified_labels does not match num of features')
#
#         # Mask for positive pairs in SupCon based on modified_labels
#         # sup_con_mask[i,j]=1 if modified_labels[i] == modified_labels[j]
#         sup_con_mask = torch.eq(current_labels_for_supcon, current_labels_for_supcon.T).float()
#
#         # Compute SupCon logits
#         # anchor_dot_contrast_supcon = torch.matmul(features, features.T) / self.temperature # Simplified
#         anchor_dot_contrast_supcon = torch.div(
#             torch.matmul(features, features.T),  # features are already anchors and contrast features
#             self.temperature
#         )
#
#         # For numerical stability
#         logits_max_supcon, _ = torch.max(anchor_dot_contrast_supcon, dim=1, keepdim=True)
#         logits_supcon = anchor_dot_contrast_supcon - logits_max_supcon.detach()
#
#         # Mask-out self-contrast cases for SupCon
#         diag_mask_supcon = torch.eye(batch_size, device=device, dtype=torch.bool)
#         logits_mask_supcon = ~diag_mask_supcon  # Use boolean mask for clarity
#
#         # Positive pairs mask for SupCon (excluding self)
#         # Only consider pairs where labels are same AND it's not a self-comparison
#         positive_mask_supcon = sup_con_mask.bool() & logits_mask_supcon
#
#         # Compute SupCon log_prob
#         # Denominator for log_prob: sum of exp(logits) over all non-self contrast pairs
#         exp_logits_supcon = torch.exp(logits_supcon)
#         exp_logits_supcon_masked = exp_logits_supcon * logits_mask_supcon  # Zero out self-contrast terms
#         log_prob_denominator = torch.log(exp_logits_supcon_masked.sum(1, keepdim=True) + 1e-9)  # Add epsilon
#
#         log_prob_supcon = logits_supcon - log_prob_denominator
#
#         # Compute SupCon loss per anchor
#         # Sum log_prob for positive pairs for each anchor, then average if positives exist
#         # (positive_mask_supcon * log_prob_supcon) will be zero for non-positive pairs
#         sum_log_prob_pos_supcon = (positive_mask_supcon.float() * log_prob_supcon).sum(1)
#         num_positives_per_anchor = positive_mask_supcon.sum(1).float()
#
#         # Avoid division by zero if an anchor has no other positive samples in the batch
#         mean_log_prob_pos_supcon = sum_log_prob_pos_supcon / (num_positives_per_anchor + 1e-9)
#
#         # Final SupCon loss
#         loss_supcon = - (self.temperature / self.base_temperature) * mean_log_prob_pos_supcon
#         loss_supcon = loss_supcon.mean()  # Average over batch
#
#         # --- Part 2: Auxiliary Loss to Pull Poisoned Samples Towards Clean Target Features ---
#         loss_pull_to_clean = torch.tensor(0.0, device=device)
#
#         # Proceed only if strength > 0 and necessary labels are provided
#         if self.pull_to_clean_strength > 0 and \
#                 original_labels is not None and \
#                 modified_labels is not None and \
#                 target_label_for_poison is not None:
#
#             # Identify indices of poisoned samples (meta_labels == 1)
#             poisoned_indices = (meta_labels == 1).nonzero(as_tuple=False).squeeze(-1)
#
#             if poisoned_indices.numel() > 0:  # If there are poisoned samples in the batch
#                 # Identify indices of clean samples that belong to the target_label_for_poison
#                 # These are samples whose meta_label is 0 AND their modified_label (which is also their original_label)
#                 # is equal to target_label_for_poison.
#                 clean_target_indices = ((meta_labels == 0) & (modified_labels == target_label_for_poison)).nonzero(
#                     as_tuple=False).squeeze(-1)
#
#                 if clean_target_indices.numel() > 0:  # If there are clean target samples in the batch
#                     features_poisoned = features[poisoned_indices]  # [num_poisoned, feat_dim]
#                     features_clean_target = features[clean_target_indices]  # [num_clean_target, feat_dim]
#
#                     # Calculate the centroid of clean target features in this batch
#                     # Detach centroid to prevent its gradients from flowing back further than necessary for this aux loss
#                     # (though for feature_extractor, gradients will flow through features_poisoned regardless)
#                     centroid_clean_target = features_clean_target.mean(dim=0, keepdim=True).detach()  # [1, feat_dim]
#
#                     # Calculate L2 squared distance from each poisoned feature to this centroid
#                     distances_sq = torch.sum((features_poisoned - centroid_clean_target) ** 2, dim=1)  # [num_poisoned]
#
#                     loss_pull_to_clean = distances_sq.mean()  # Average distance for all processed poisoned samples
#                 else:
#                     # No clean target samples in this batch to pull towards
#                     # loss_pull_to_clean remains 0.0
#                     pass
#             else:
#                 # No poisoned samples in this batch
#                 # loss_pull_to_clean remains 0.0
#                 pass
#
#         # --- Total Loss ---
#         total_loss = loss_supcon + self.pull_to_clean_strength * loss_pull_to_clean
#
#         # For debugging:
#         # if torch.isnan(total_loss) or torch.isinf(total_loss):
#         #     logger.error(f"NaN or Inf loss detected!")
#         #     logger.error(f"Loss SupCon: {loss_supcon.item()}, Loss PullToClean: {loss_pull_to_clean.item()}")
#         #     logger.error(f"Num positives per anchor: {num_positives_per_anchor}")
#         #     # Potentially save inputs or states for debugging
#         # else:
#         #     if self.pull_to_clean_strength > 0 and meta_labels.sum() > 0 : # If expecting pull loss
#         #          logger.debug(f"BS:{batch_size}, LS:{loss_supcon.item():.4f}, LP:{loss_pull_to_clean.item():.4f}, NPois:{meta_labels.sum().item()}")
#
#         return total_loss