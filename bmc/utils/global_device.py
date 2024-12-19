import torch

GLOBAL_DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")