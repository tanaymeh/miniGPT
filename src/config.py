class Config:
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 80  # Max sequence size
    embed_dim = 256  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer