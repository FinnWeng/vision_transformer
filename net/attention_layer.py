'''
Borrow from jax implementation
The multihead attention of tensorflow is matmul version.

dot_product_attention_weights: q dot k.

then

dot_product_attention: (q dot k) dot v


'''

import tensorflow as tf
from typing import (Any, Callable, Tuple, Optional)

Shape = Tuple[int]
Dtype = Any
Array = Any


def dot_product_attention_weights(query: Array,
                                  key: Array,
                                  bias: Optional[Array] = None,
                                  dropout_rate: float = 0.,
                                  deterministic: bool = False,
                                  dtype: Dtype = tf.float32):
    """Computes dot-product attention weights given query and key.
    
    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
        query: queries for calculating attention with shape of
            `[batch..., q_length, num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of
            `[batch..., kv_length, num_heads, qk_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
            shape `[batch..., num_heads, q_length, kv_length]`.
            This can be used for incorporating causal masks, padding masks,
            proximity bias, etc.
        broadcast_dropout: bool: use a broadcasted dropout along batch dims. (no using in vit)
        # dropout_rng: JAX PRNGKey: to be used for dropout(this is for jax. no need for us)
        dropout_rate: dropout rate
        deterministic: bool, deterministic or not (to apply dropout)
        dtype: the dtype of the computation (default: float32)
        # precision: numerical precision of the computation see `jax.lax.Precision`
            for details.(no need for this implementation)

    Returns:
        Output of shape `[batch..., num_heads, q_length, kv_length]`.
    """
    assert (tf.rank(query) == tf.rank(key)).numpy(), 'q, k must have same rank.'
    assert query.shape[:-3] == key.shape[:-3], ('q, k batch dims must match.')
    assert query.shape[-2] == key.shape[-2], ('q, k num_heads must match.')
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / tf.math.sqrt(tf.cast(depth,dtype))
      # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = tf.einsum('...qhd,...khd->...hqk', query, key)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias

    # normalize the attention weights
    attn_weights = tf.keras.layers.Softmax(attn_weights).astype(dtype)

    # apply attention dropout
    if not deterministic and dropout_rate > 0.:
        # keep_prob = 1.0 - dropout_rate
        # if broadcast_dropout:
        #     # dropout is broadcast across the batch + head dimensions
        #     dropout_shape = tuple([1] * (tf.rank(key).numpy() - 2)) + attn_weights.shape[-2:]
        #     keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        # else:
        #     keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        # multiplier = (keep.astype(attn_weights.dtype) /
        #             jnp.asarray(keep_prob, dtype=dtype))
        # attn_weights = attn_weights * multiplier

        attn_weights = tf.keras.layers.Dropout(dropout_rate)(attn_weights)


    return attn_weights
    
def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: Dtype = tf.float32):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Note: query, key, value needn't have any batch dimensions.

    Args:
        query: queries for calculating attention with shape of
            `[batch..., q_length, num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of
            `[batch..., kv_length, num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of
            `[batch..., kv_length, num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
            shape `[batch..., num_heads, q_length, kv_length]`.
            This can be used for incorporating causal masks, padding masks,
            proximity bias, etc.
        # broadcast_dropout: bool: use a broadcasted dropout along batch dims.(not using)
        # dropout_rng: JAX PRNGKey: to be used for dropout(for jax)
        dropout_rate: dropout rate
        deterministic: bool, deterministic or not (to apply dropout)
        dtype: the dtype of the computation (default: float32)
        # precision: numerical precision of the computation see `jax.lax.Precision`
            for details(not using).

    Returns:
        Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
    """
    assert (tf.rank(key) == tf.rank(query) == tf.rank(value)).numpy(), 'q, k, v must have same rank.'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        'q, k, v batch dims must match.')
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        'q, k, v num_heads must match.')
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

    # compute attention weights
    attn_weights = dot_product_attention_weights(query, key, bias, dropout_rate,deterministic, dtype)

    # return weighted sum over values for each query position
    return tf.einsum('...hqk,...khd->...qhd', attn_weights, value)

class Multi_Head_Dot_Product_Attention(f.keras.layers.Layer):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
    """
    def __init__(self,
        num_heads: int,
        dtype: Dtype = jnp.float32,
        qkv_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout_rate: float = 0.,
        deterministic: Optional[bool] = None,
        kernel_init: Callable = tf.keras.initializers.GlorotUniform,
        bias_init: Callable = tf.keras.initializers.Zeros,
        use_bias: bool = True,
        attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention,
        decode: bool = False,
        ) -> None:
        self.num_heads = num_heads,
        self.dtype = dtype,
        self.qkv_features = qkv_features,
        self.out_features = out_features,
        self.dropout_rate = dropout_rate,
        self.deterministic = deterministic,
        self.kernel_init = kernel_init,
        self.bias_init = bias_init,
        self.use_bias = use_bias,
        self.attention_fn = attention_fn,
        self.decode = decode

    def call(self,
            inputs_q: Array,
            inputs_kv: Array,
            mask: Optional[Array] = None,
            deterministic: Optional[bool] = None):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
        inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
        inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
        mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
        deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
        output of shape `[batch_sizes..., length, features]`.
        """
        if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
            self.deterministic = deterministic
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, ('Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads
