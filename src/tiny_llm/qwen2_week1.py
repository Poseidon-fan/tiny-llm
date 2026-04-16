import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = hidden_size // num_heads

        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.rope = RoPE(hidden_size // num_heads, max_seq_len, theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = linear(x, self.wq, self.bq)
        k = linear(x, self.wk, self.bk)
        v = linear(x, self.wv, self.bv)

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        x = scaled_dot_product_attention_grouped(
            q.astype(mx.float32), 
            k.astype(mx.float32), 
            v.astype(mx.float32), 
            scale=mx.rsqrt(self.head_dim), 
            mask=mask
        )

        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(x, self.wo)

class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        gate = silu(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        return linear(gate * up, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.attention = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq, wk=wk, wv=wv, wo=wo,
            bq=bq, bk=bk, bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.mlp = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, eps=rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, mask=mask)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual

        return x


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        args = mlx_model.args
        model = mlx_model.model

        self.embedding = Embedding(
            args.vocab_size,
            args.hidden_size,
            dequantize_linear(model.embed_tokens),
        )

        self.layers = []
        for layer in model.layers:
            self.layers.append(Qwen2TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                wq=dequantize_linear(layer.self_attn.q_proj),
                wk=dequantize_linear(layer.self_attn.k_proj),
                wv=dequantize_linear(layer.self_attn.v_proj),
                wo=dequantize_linear(layer.self_attn.o_proj),
                bq=layer.self_attn.q_proj.bias,
                bk=layer.self_attn.k_proj.bias,
                bv=layer.self_attn.v_proj.bias,
                w_gate=dequantize_linear(layer.mlp.gate_proj),
                w_up=dequantize_linear(layer.mlp.up_proj),
                w_down=dequantize_linear(layer.mlp.down_proj),
                w_input_layernorm=layer.input_layernorm.weight,
                w_post_attention_layernorm=layer.post_attention_layernorm.weight,
                max_seq_len=args.max_position_embeddings,
                theta=args.rope_theta,
            ))

        self.norm = RMSNorm(args.hidden_size, model.norm.weight, eps=args.rms_norm_eps)

        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = dequantize_linear(mlx_model.lm_head)

    def __call__(self, inputs: mx.array) -> mx.array:
        x = self.embedding(inputs)

        L = x.shape[1]
        mask = "causal" if L > 1 else None

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)

        if self.lm_head is None:
            return self.embedding.as_linear(x)
        else:
            return linear(x, self.lm_head)
