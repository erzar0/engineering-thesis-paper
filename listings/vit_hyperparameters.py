input_size = 4096
patch_size = 16 
embed_dim = 16 
num_heads = 8
num_classes = 20
num_layers = 6
hidden_dim = 4 * embed_dim
dropout_rate = 0.1

model = VisionTransformer(input_size, patch_size, embed_dim, num_heads, num_classes, num_layers, hidden_dim, dropout_rate)
