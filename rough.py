import torch

# we need mean and variance
# mean = sqrt(alpha_hat_t)*x_0
# variance = sqrt(1-alpha_hat_t)*random_noise

x0 = torch.randn(2, 3, 32, 32) #random tensor, 2 samples, 3 channels, 32x32 pixels

betas = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25]) #5 betas

# timestamp
t = torch.tensor([1, 3])

# as betas are amount of noise that are applied at every timestamp
# alhpas = 1-betas = amount of original information that is been preserved at every timestamp


alphas = 1 - betas

alpha_hat = torch.cumprod(alphas, axis=0) #cumulative product of alphas

result = alpha_hat.gather(-1, t).reshape(-1, 1, 1, 1)

noise = torch.randn_like(x0)

mean = result.sqrt() * x0
variance = torch.sqrt(1 - result) * noise

x_t = mean + variance


