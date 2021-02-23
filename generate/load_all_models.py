import os
import torch
from torchvision import utils
from model import Generator
import matplotlib.pyplot as plt
import time
import numpy as np

path_models = '/home/morzh/work/stylegan2-pytorch/models'

file_1024_new = 'stylegan-1024px-new.model'
file_512_new = 'stylegan-512px-new.model'
file_512 = 'stylegan-1024px.model'
file_generator_ffthq = 'generator_ffhq.pt'

device = torch.device("cpu")

# model_1024_new = torch.load(os.path.join(path_models, file_1024_new), map_location=device)
# model_1024_new.eval()

g_ema_ffthq = Generator(256, 512, 0).to(device)
model_gengerator_ffthq = torch.load(os.path.join(path_models, file_generator_ffthq), map_location=device)
g_ema_ffthq.load_state_dict(model_gengerator_ffthq["g_ema"], strict=False)
g_ema_ffthq.eval()
