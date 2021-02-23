import os
import torch
from torchvision import utils
from model import Generator
import matplotlib.pyplot as plt
import time
import numpy as np


latent = 512
n_mlp = 8
size = 1024
channel_multiplier = 2
truncation = 1
truncation_mean = 4096
rand_sample = 1
num_samples = 100

# ckpt = '/home/morzh/work/stylegan2-pytorch/models/stylegan-1024px-new.model'
ckpt = '/home/morzh/work/stylegan2-pytorch/checkpoint/stylegan2-ffhq-config-f.pt'

faces_path = '/media/morzh/ext4_volume/data/Faces/BeautifyMeFaceset-005/01_Neutral'
faces_filename = '000002.npy'
faces_filepath = os.path.join(faces_path, faces_filename)
face = np.load(faces_filepath)
face = np.expand_dims(face, axis=0)
face = torch.from_numpy(face)
faces = [face]

device = torch.device("cpu")
checkpoint = torch.load(ckpt, map_location=device)
g_ema = Generator(size, latent, n_mlp, channel_multiplier=channel_multiplier).to(device)

# g_ema.load_state_dict(checkpoint['generator'])
g_ema.load_state_dict(checkpoint["g_ema"])

if truncation < 1:
    with torch.no_grad():
        mean_latent = g_ema.mean_latent(truncation_mean)
else:
    mean_latent = None


with torch.no_grad():
    g_ema.eval()
    for face in faces:
        time_start = time.time()
        sample, _ = g_ema([face], truncation=truncation, truncation_latent=mean_latent, input_is_latent=True)
        time_end = time.time()

        print('time for inference is', time_end-time_start, 'seconds')
        img = sample[0].permute(1, 2, 0)
        img = (img + 1.0) / 2.0
        plt.figure(figsize=(20, 10))
        plt.imshow(img)
        plt.tight_layout()
        plt.show()