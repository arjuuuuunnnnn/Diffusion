import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import math
import matplotlib.pyplot as plt
import torch
import urllib
import numpy as np
import PIL
from unet import UNet

device = 'cpu'

def get_sample_image()->PIL.Image.Image:
    url = ""
    filename = "sample.jpg"
    urllib.request.urlretrieve(url, filename)
    return PIL.Image.open(filename)

class DiffusionModel:
    def __init__(self, start_schedule=0.001, end_schedule=0.02, timesteps=1000):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps

        self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        

    def forward_diffusion(x0, t, betas = torch.linspace(0, 1, 5)):
        noise = torch.rand_like(x0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x0.shape)

        mean = sqrt_alphas_cumprod_t.to(device) * x0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

        xt = mean + variance
        
        return xt, noise.to(device)

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        result = values.gather(-1, t.cpu())

        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


        
