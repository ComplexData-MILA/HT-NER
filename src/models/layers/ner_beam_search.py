from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

def create_ner_conditional_masks(id2label: Dict[int, str]) -> torch.Tensor:
    """Create a NER-conditional mask matrix which implies the relations between
    before-tag and after-tag.
    According to the rule of BIO-naming system, it is impossible that `I-Dog` cannot be
    appeard after `B-Dog` or `I-Dog` tags. This function creates the calculable
    relation-based conditional matrix to prevent from generating wrong tags.
    Args:
        id2label: A dictionary which maps class indices to their label names.
    Returns:
        A conditional mask tensor.
    """
    conditional_masks = torch.zeros(len(id2label), len(id2label))
    for i, before in id2label.items():
        for j, after in id2label.items():
            if after == "O" or after.startswith("B-") or after == f"I-{before[2:]}":
                conditional_masks[i, j] = 1.0
    return conditional_masks


# source: https://github.com/affjljoo3581/Feedback-Prize-Competition/blob/master/src/utils/ner_utils.py
def ner_beam_search_decode(
    log_probs: torch.Tensor, id2label: Dict[int, str], beam_size: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode NER-tags from the predicted log-probabilities using beam-search.
    This function decodes the predictions using beam-search algorithm. Because all tags
    are predicted simultaneously while the tags have dependencies of their previous
    tags, the greedy algorithm cannot decode the tags properly. With beam-search, it is
    possible to prevent the below situation:
        >>> sorted = probs[t].sort(dim=-1)
        >>> print("\t".join([f"{id2label[i]} {p}" for p, i in zip()]))
        I-Dog 0.54  B-Cat 0.44  ...
        >>> sorted = probs[t + 1].sort(dim=-1)
        >>> print("\t".join([f"{id2label[i]} {p}" for p, i in zip()]))
        I-Cat 0.99  I-Dog 0.01  ...
        >>> # Decode the NER-tags with beam-search algorithm.
        >>> preds, pred_probs = ner_beam_search_decode(
        ...    concat_tensors_with_padding(logits, padding=0).float().log_softmax(dim=-1),
        ...    self.id2label,
        ...    self.config.model.decoding.beam_size,
        ... )
    The above shows that if the locally-highest tags are selected, then `I-Dog, I-Dog`
    will be generated even the confidence of the second tag `I-Dog` is significantly
    lower than `I-Cat`. It is more natural that `B-Cat, I-Cat` is generated rather than
    `I-Dog, I-Dog`. The beam-search for NER-tagging task can solve this problem.
    Args:
        log_probs: The log-probabilities of the token predictions.
        id2label: A dictionary which maps class indices to their label names.
        beam_size: The number of candidates for each search step. Default is `2`.
    Returns:
        A tuple of beam-searched indices and their probability tensors.
    """
    # Create the log-probability mask for the invalid predictions.
    log_prob_masks = -10000.0 * (1 - create_ner_conditional_masks(id2label))
    log_prob_masks = log_prob_masks.to(log_probs.device)

    beam_search_shape = (log_probs.size(0), beam_size, log_probs.size(1))
    searched_tokens = log_probs.new_zeros(beam_search_shape, dtype=torch.long)
    searched_log_probs = log_probs.new_zeros(beam_search_shape)

    searched_scores = log_probs.new_zeros(log_probs.size(0), beam_size)
    searched_scores[:, 1:] = -10000.0

    for i in range(log_probs.size(1)):
        # Calculate the accumulated score (log-probabilities) with excluding invalid
        # next-tag predictions.
        scores = searched_scores.unsqueeze(2)
        scores = scores + log_probs[:, i, :].unsqueeze(1)
        scores = scores + (log_prob_masks[searched_tokens[:, :, i - 1]] if i > 0 else 0)

        # Select the top-k (beam-search size) predictions.
        best_scores, best_indices = scores.flatten(1).topk(beam_size)
        best_tokens = best_indices % scores.size(2)
        best_log_probs = log_probs[:, i, :].gather(dim=1, index=best_tokens)

        best_buckets = best_indices.div(scores.size(2), rounding_mode="floor")
        best_buckets = best_buckets.unsqueeze(2).expand(-1, -1, log_probs.size(1))

        # Gather the best buckets and their log-probabilities.
        searched_tokens = searched_tokens.gather(dim=1, index=best_buckets)
        searched_log_probs = searched_log_probs.gather(dim=1, index=best_buckets)

        # Update the predictions by inserting to the corresponding timestep.
        searched_scores = best_scores
        searched_tokens[:, :, i] = best_tokens
        searched_log_probs[:, :, i] = best_log_probs

    # Return the best beam-searched sequence and its probabilities.
    return searched_tokens[:, 0, :], searched_log_probs[:, 0, :].exp()