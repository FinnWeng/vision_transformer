import ml_collections

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'ViT-B_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    # config.hidden_size = 768
    config.hidden_size = 128
    config.transformer = ml_collections.ConfigDict()
    # config.transformer.mlp_dim = 3072
    config.transformer.mlp_dim = 128
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.name = 'ViT-B_32'
    config.patches.size = (32, 32)
    return config