import torch.nn as nn
from torch.nn import init

def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
