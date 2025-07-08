import torch

EPS = 1e-8

Z_VALUE_CAP = 5.0 #(random) suggestion by an LLM 

def stablize(z):
    z = z + EPS * torch.randn_like(z)
    z = torch.nan_to_num(z, nan=0.0, posinf=Z_VALUE_CAP, neginf=-Z_VALUE_CAP)
    z = torch.clamp(z, -Z_VALUE_CAP, Z_VALUE_CAP)
    return z