import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        q = linear(query, self.wq)
        k = linear(key, self.wk)
        v = linear(value, self.wv)

        q = q.reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        x = scaled_dot_product_attention_simple(q, k, v, mask=mask)
        x = x.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(x, self.wo)



def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, 0, -mx.inf)
    return mask.astype(dtype)



def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    B = query.shape[:-3]
    n_repeats = H_q // H
    
    query = query.reshape(*B, H, n_repeats, L, D)

    key   = key.reshape(*B, H, 1, S, D)
    value = value.reshape(*B, H, 1, S, D)

    factor = mx.rsqrt(D) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, scores.dtype)
        else:
            mask = mask.reshape(*B, H, n_repeats, L, S)
        scores = scores + mask
    result = mx.matmul(softmax(scores, axis=-1), value)
    return result.reshape(*B, H_q, L, D)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
