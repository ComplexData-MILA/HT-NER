import torch

def unsorted_segment_sum(data, segment_ids, num_segments, x=None):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    segment_ids = torch.repeat_interleave(segment_ids.unsqueeze(-1),
                                            repeats=data.shape[-1],
                                            dim=-1)
    shape = [data.shape[0], num_segments] + list(data.shape[2:])
    if x is None:
        x = torch.zeros(*shape, dtype=data.dtype, device=data.device)
    x.scatter_add_(1, segment_ids, data)
    return x

def unsorted_segment_mean(data, segment_ids, num_segments):
    """
    Computes the mean along segments of a tensor. Analogous to tf.unsorted_segment_mean.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    tensor = unsorted_segment_sum(data, segment_ids, num_segments)
    base = unsorted_segment_sum(torch.ones_like(data), segment_ids, num_segments)
    ## base + 1e-5 lb 693, clamp 690 not sure if by randomness
    base = torch.clamp(base, min=1.)
    # base += 1e-5
    tensor = tensor / base
    return tensor

def unsorted_segment_reduce(data, segment_ids, num_segments, combiner='sum'):
    if combiner == 'sum':
        return unsorted_segment_sum(data, segment_ids, num_segments)
    elif combiner == 'mean' or combiner == 'avg':
        return unsorted_segment_mean(data, segment_ids, num_segments)

def groupby(logits, word_ids, combiner='sum', max_words=28118):
    """
    >>> x = backbone(input_ids=input_ids, attention_mask=attention_mask)[0]
    >>> # merge from tokens to words
    >>> x = groupby(x, word_ids, combiner=FLAGS.word_combiner, max_words=max_words) 
    >>> x, _ = lstm(x)
    """
    word_ids_ = word_ids + 1
    word_ids_ *= (word_ids_ < max_words).long()
    logits = unsorted_segment_reduce(logits, word_ids_.long(), max_words + 1, combiner=combiner)
    return logits[:, 1:]

