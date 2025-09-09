import torch

def denorm_for_clip(x):
    # x: [C,H,W] tensor normalized with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    return (x * std + mean).clamp(0, 1)
