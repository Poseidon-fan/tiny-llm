import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        i = mx.arange(0, dims // 2)
        theta = 1.0 / (base ** (2 * i / dims))

        positions = mx.arange(seq_len)
        freqs = mx.outer(positions, theta)

        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        L = x.shape[1]
        if offset is None:
            cos = self.cos_freqs[:L, None, :]
            sin = self.sin_freqs[:L, None, :]
        else:
            cos = self.cos_freqs[offset, None, :]
            sin = self.sin_freqs[offset, None, :]
        
        if self.traditional:
            x_pair = x.reshape(*x.shape[:-1], -1, 2)
            out0 = x_pair[..., 0] * cos - x_pair[..., 1] * sin
            out1 = x_pair[..., 0] * sin + x_pair[..., 1] * cos
            return mx.stack([out0, out1], axis=-1).reshape(*x.shape)
        else:
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            out0 = x1 * cos - x2 * sin
            out1 = x1 * sin + x2 * cos
            return mx.concatenate([out0, out1], axis=-1)

