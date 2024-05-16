import torch.nn as nn

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count:,} trainable parameters')
    return count

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
