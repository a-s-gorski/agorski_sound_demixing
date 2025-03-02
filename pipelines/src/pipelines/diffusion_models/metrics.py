import torch

def sdr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s_target = torch.norm(target, dim=-1)**2 + eps
    s_error = torch.norm(target - preds, dim=-1)**2 + eps
    return 10 * torch.log10(s_target/s_error)


def sisnr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # safeguard for varied len samples
    s_len = min(preds.shape[1], target.shape[1])
    preds = preds[:, :s_len]
    target = target[:, :s_len]
    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    target_scaled = alpha * target
    noise = target_scaled - preds
    s_target = torch.sum(target_scaled**2, dim=-1) + eps
    s_error = torch.sum(noise**2, dim=-1) + eps
    return 10 * torch.log10(s_target / s_error)