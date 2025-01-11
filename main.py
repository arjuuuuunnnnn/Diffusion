import torch

def forward_diffusion(x0, t, betas = torch.linspace(0, 1, 5)):
    noise = torch.rand_like(x0)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, axis=0)
    alpha_hat_t = alpha_hat.gather(-1, t).reshape(-1, 1, 1, 1)

    mean = alpha_hat_t.sqrt() * x0
    variance = torch.sqrt(1-alpha_hat_t) * noise

    return mean+variance, noise


