import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        return (x * rms * self.weight).astype(x.dtype)
