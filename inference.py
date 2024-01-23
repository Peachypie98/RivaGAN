import gc
import json
import os
import cv2
import torch
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_dct as dct
import random
import time
import argparse

from tqdm import tqdm
from itertools import chain
from torch import optim
from torch.nn import functional
from math import exp
from torch.nn.functional import conv2d
from glob import glob
from torch.utils.data import DataLoader, Dataset

#####################################################################################
# Initialization                                                                    #
#####################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings('ignore')

#####################################################################################
# Argument Parser                                                                   #
#####################################################################################
parser = argparse.ArgumentParser() 
parser.add_argument('--data_dim', type=int, default=32)
parser.add_argument('--model_weight', type=str, default=None)
parser.add_argument('--random_data', type=str, default="Yes")
parser.add_argument('--your_data', type=str, default=None)
parser.add_argument('--video_location', type=str, default="./data/hollywood2/val/actioncliptest00002.avi")
parser.add_argument('--fps', type=int, default=25)
args = parser.parse_args()

#####################################################################################
# Model                                                                             #
#####################################################################################
class Critic(nn.Module):
    """
    The Critic module maps a video to a scalar score. It takes in a batch of N videos - each
    of which is of length L, height H, and width W - and produces a score for each video which
    corresponds to how "realistic" it looks.

    Input: (N, 3, L, H, W)
    Output: (N, 1)
    """

    def __init__(self, kernel_size=(1, 3, 3), padding=(0, 0, 0)):
        super(Critic, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=kernel_size, padding=padding, stride=2),
            nn.Tanh(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=kernel_size, padding=padding, stride=2),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=kernel_size, padding=padding, stride=2),
        )
        self._linear = nn.Linear(64, 1)

    def forward(self, frames):
        frames = self._conv(frames)
        N, _, L, H, W = frames.size()
        return self._linear(torch.mean(frames.view(N, -1, L * H * W), dim=2))

class Adversary(nn.Module):
    """
    The Adversary module maps a sequence of frames to another sequence of frames
    with a constraint on the maximum distortion of each individual pixel.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, l1_max=0.05, kernel_size=(1, 3, 3), padding=(0, 1, 1)):
        super(Adversary, self).__init__()
        self.l1_max = l1_max
        self._conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 3, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
        )

    def forward(self, frames):
        x = frames
        x = self._conv(x)
        return frames + self.l1_max * x

def multiplicative(x, data):
    """
    This function takes a 5d tensor (with the same shape and dimension order
    as the input to Conv3d) and a 2d data tensor. For each element in the
    batch, the data vector is combined with the first D dimensions of the 5d
    tensor through an elementwise product.

    Input: (N, C_{in}, L, H, W), (N, D)
    Output: (N, C_{in}, L, H, W)
    """
    N, D = data.size()
    N, C, L, H, W = x.size()
    assert D <= C, "data dims must be less than channel dims"
    x = torch.cat([
        x[:, :D, :, :, :] * data.view(N, D, 1, 1, 1).expand(N, D, L, H, W),
        x[:, D:, :, :, :]
    ], dim=1)
    return x

class AttentiveEncoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D, )
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim, tie_rgb=False, linf_max=0.016,
                 kernel_size=(1, 11, 11), padding=(0, 5, 5)):
        super(AttentiveEncoder, self).__init__()

        self.linf_max = linf_max
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.padding = padding

        self._attention = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=kernel_size, padding=padding), # [3,3,2,H,W]
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, data_dim, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.BatchNorm3d(data_dim),
        )
        self._conv = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 1 if tie_rgb else 3, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        data = data * 2.0 - 1.0
        x = functional.softmax(self._attention(frames), dim=1)
        x = torch.sum(multiplicative(x, data), dim=1, keepdim=True)
        x = self._conv(torch.cat([frames, x], dim=1))
        return frames + self.linf_max * x

class AttentiveDecoder(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, D)
    """

    def __init__(self, encoder):
        super(AttentiveDecoder, self).__init__()
        self.data_dim = encoder.data_dim
        self._attention = encoder._attention
        self._conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=encoder.kernel_size, padding=encoder.padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, self.data_dim, kernel_size=encoder.kernel_size,
                      padding=encoder.padding, stride=1),
        )

    def forward(self, frames):
        N, D, L, H, W = frames.size()
        attention = functional.softmax(self._attention(frames), dim=1)
        x = self._conv(frames) * attention
        return torch.mean(x.view(N, self.data_dim, -1), dim=2)

class Crop(nn.Module):
    """
    Randomly crops the two spatial dimensions independently to a new size
    that is between `min_pct` and `max_pct` of the old size.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H', W')
    """

    def __init__(self, min_pct=0.8, max_pct=1.0):
        super(Crop, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def _pct(self):
        return self.min_pct + random.random() * (self.max_pct - self.min_pct)

    def forward(self, frames):
        _, _, _, height, width = frames.size()
        dx = int(self._pct() * width)
        dy = int(self._pct() * height)
        dx, dy = (dx // 4) * 4, (dy // 4) * 4
        x = random.randint(0, width - dx - 1)
        y = random.randint(0, height - dy - 1)
        return frames[:, :, :, y:y + dy, x:x + dx]

class Scale(nn.Module):
    """
    Randomly scales the two spatial dimensions independently to a new size
    that is between `min_pct` and `max_pct` of the old size.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_pct=0.8, max_pct=1.0):
        super(Scale, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def _pct(self):
        return self.min_pct + random.random() * (self.max_pct - self.min_pct)

    def forward(self, frames):
        percent = self._pct()
        _, _, depth, height, width = frames.size()
        height, width = int(percent * height), int(percent * width)
        height, width = (height // 4) * 4, (width // 4) * 4
        return nn.AdaptiveAvgPool3d((depth, height, width))(frames)

class Compression(nn.Module):
    """
    This uses the DCT to produce a differentiable approximation of JPEG compression.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, yuv=False, min_pct=0.0, max_pct=0.5):
        super(Compression, self).__init__()
        self.yuv = yuv
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, y):
        N, _, L, H, W = y.size()

        L = int(y.size(2) * (random.random() * (self.max_pct - self.min_pct) + self.min_pct))
        H = int(y.size(3) * (random.random() * (self.max_pct - self.min_pct) + self.min_pct))
        W = int(y.size(4) * (random.random() * (self.max_pct - self.min_pct) + self.min_pct))

        if self.yuv:
            y = torch.stack([
                (0.299 * y[:, 2, :, :, :] +
                 0.587 * y[:, 1, :, :, :] +
                 0.114 * y[:, 0, :, :, :]),
                (- 0.168736 * y[:, 2, :, :, :] -
                 0.331264 * y[:, 1, :, :, :] +
                 0.500 * y[:, 0, :, :, :]),
                (0.500 * y[:, 2, :, :, :] -
                 0.418688 * y[:, 1, :, :, :] -
                 0.081312 * y[:, 0, :, :, :]),
            ], dim=1)

        y = dct.dct_3d(y)

        if L > 0:
            y[:, :, -L:, :, :] = 0.0

        if H > 0:
            y[:, :, :, -H:, :] = 0.0

        if W > 0:
            y[:, :, :, :, -W:] = 0.0

        y = dct.idct_3d(y)

        if self.yuv:
            y = torch.stack([
                (1.0 * y[:, 0, :, :, :] +
                 1.772 * y[:, 1, :, :, :] +
                 0.000 * y[:, 2, :, :, :]),
                (1.0 * y[:, 0, :, :, :] -
                 0.344136 * y[:, 1, :, :, :] -
                 0.714136 * y[:, 2, :, :, :]),
                (1.0 * y[:, 0, :, :, :] +
                 0.000 * y[:, 1, :, :, :] +
                 1.402 * y[:, 2, :, :, :]),
            ], dim=1)

        return y

#####################################################################################
# Utils                                                                             #
#####################################################################################
def gaussian(window_size, sigma):
    """Gaussian window.

    https://en.wikipedia.org/wiki/Window_function#Gaussian_window
    """
    _exp = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.Tensor(_exp)
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):

    padding_size = window_size // 2

    mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    _ssim_quotient = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    _ssim_divident = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = _ssim_quotient / _ssim_divident

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def mjpeg(x):
    """
    Write each video to disk and re-read it from disk.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """
    y = torch.zeros(x.size())
    _, _, _, height, width = x.size()

    for n in range(x.size(0)):
        vout = cv2.VideoWriter(f"./temporary_files/tmp.avi", cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (width, height))
        for l in range(x.size(2)):
            image = x[n, :, l, :, :]  # (3, H, W)
            image = torch.clamp(image.permute(1, 2, 0), min=-1.0, max=1.0)
            vout.write(((image + 1.0) * 127.5).detach().cpu().numpy().astype("uint8"))
        vout.release()

        
        vin = cv2.VideoCapture(f"./temporary_files/tmp.avi")
        for l in range(x.size(2)):
            _, frame = vin.read()  # (H, W, 3)
            frame = torch.tensor(frame / 127.5 - 1.0)
            y[n, :, l, :, :] = frame.permute(2, 0, 1)
    return y.to(x.device)

#########################################################################################
# Dataloader                                                                            #
#########################################################################################
class VideoDataset(Dataset):
    """
    Given a folder of *.avi video files organized as shown below, this dataset
    selects randomly crops the video to `crop_size` and returns a random
    continuous sequence of `seq_len` frames of shape.

        /root_dir
            1.avi
            2.avi

    The output has shape (3, seq_len, crop_size[0], crop_size[1]).
    """

    def __init__(self, root_dir, crop_size, seq_len, max_crop_size=(360, 480)):
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.max_crop_size = max_crop_size

        self.videos = []
        for ext in ["avi", "mp4"]:
            for path in glob(os.path.join(root_dir, "**/*.%s" % ext), recursive=True):
                cap = cv2.VideoCapture(path)
                nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.videos.append((path, nb_frames))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # Select time index
        path, nb_frames = self.videos[idx]
        start_idx = random.randint(0, nb_frames - self.seq_len - 1) 

        # Select space index
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx - 1) # read frame from start_idx-1
        ok, frame = cap.read()
        H, W, D = frame.shape
        x, dx, y, dy = 0, W, 0, H
        if self.crop_size:
            dy, dx = self.crop_size
            x = random.randint(0, W - dx - 1)
            y = random.randint(0, H - dy - 1)
        if self.max_crop_size[0] < dy:
            dy, dx = self.max_crop_size
            y = random.randint(0, H - dy - 1)
        if self.max_crop_size[1] < dx:
            dy, dx = self.max_crop_size
            x = random.randint(0, W - dx - 1)

        # Read frames and normalize to [-1.0, 1.0] 
        frames = []
        for _ in range(self.seq_len): # read specific number of frames
            ok, frame = cap.read() # read frame from start_idx-1
            frame = frame[y:y + dy, x:x + dx]
            frames.append(frame / 127.5 - 1.0) 
        x = torch.FloatTensor(frames)
        x = x.permute(3, 0, 1, 2) # [C, T, H, W]
        return x

def load_train_val(seq_len, batch_size, dataset="hollywood2"):
    """
    This returns two dataloaders correponding to the train and validation sets. Each
    iterator yields tensors of shape (N, 3, L, H, W) where N is the batch size, L is
    the sequence length, and H and W are the height and width of the frame.

    The batch size is always 1 in the validation set. The frames are always cropped
    to (128, 128) --> (160,160) windows in the training set. The frames in the validation set are
    not cropped if they are smaller than 360x480; otherwise, they are cropped so the
    maximum returned size is 360x480.
    """
    train = DataLoader(VideoDataset(
        "%s/train" % dataset,
        crop_size=(160, 160),
        seq_len=seq_len,
    ), shuffle=True, num_workers=16, batch_size=batch_size, pin_memory=True)
    val = DataLoader(VideoDataset(
        "%s/val" % dataset,
        crop_size=False,
        seq_len=seq_len,
    ), shuffle=False, batch_size=1, pin_memory=True)
    return train, val

#########################################################################################
# Train                                                                                 #
#########################################################################################
def get_acc(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()

def quantize(frames):
    # [-1.0, 1.0] -> {0, 255} -> [-1.0, 1.0]
    return ((frames + 1.0) * 127.5).int().float() / 127.5 - 1.0

def make_pair(frames, data_dim, use_bit_inverse=True, multiplicity=1):
    # Add multiplicity to further stabilize training.
    frames = torch.cat([frames] * multiplicity, dim=0).cuda() 
    if random.random() > 0.5:
        mylist = ["0", "1"]
        data = torch.zeros((frames.size(0)-1, data_dim)).random_(0, 2)
        if random.choice(mylist) == "0":
            data2 = torch.zeros((1, data_dim))
            data = torch.concat((data,data2), dim=0).cuda()
        else:   
            data2 = torch.ones((1, data_dim))
            data = torch.concat((data,data2), dim=0).cuda()
    else:
        data = torch.zeros((frames.size(0), data_dim)).random_(0, 2).cuda()

    # Add the bit-inverse to stabilize training.
    if use_bit_inverse:
        frames = torch.cat([frames, frames], dim=0).cuda()
        data = torch.cat([data, 1.0 - data], dim=0).cuda()
    return frames, data

class RivaGAN(object):
    def __init__(self, model="attention", data_dim=32):
        self.model = model
        self.data_dim = data_dim
        self.adversary = Adversary().cuda()
        self.critic = Critic().cuda()
        if model == "attention":
            self.encoder = AttentiveEncoder(data_dim=data_dim).cuda()
            self.decoder = AttentiveDecoder(self.encoder).cuda()
        else:
            raise ValueError("Unknown model: %s" % model)

    def fit(self, dataset, log_dir=False, 
            seq_len=1, batch_size=12, lr=5e-4,
            use_critic=True, use_adversary=True,
            epochs=300, use_bit_inverse=True, use_noise=True):

        if not log_dir:
            log_dir = "experiments/%s-%s" % (self.model, str(int(time.time())))
        os.makedirs(log_dir, exist_ok=False)

        # Set up the noise layers
        crop = Crop()
        scale = Scale()
        compress = Compression()

        def noise(frames):
            if use_noise:
                if random.random() < 0.5:
                    frames = crop(frames)
                if random.random() < 0.5:
                    frames = scale(frames)
                if random.random() < 0.5:
                    frames = compress(frames)
            return frames

        # Set up the data and optimizers
        train, val = load_train_val(seq_len, batch_size, dataset)
        G_opt = optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr)
        G_scheduler = optim.lr_scheduler.ReduceLROnPlateau(G_opt)
        D_opt = optim.Adam(chain(self.adversary.parameters(), self.critic.parameters()), lr=lr)
        # D_scheduler = optim.lr_scheduler.ReduceLROnPlateau(D_opt)

        # Set up the log directory
        with open(os.path.join(log_dir, "config.json"), "wt") as fout:
            fout.write(json.dumps({
                "model": self.model,
                "data_dim": self.data_dim,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dataset": dataset,
                "lr": lr,
                "log_dir": log_dir,
            }, indent=2, default=lambda o: str(o)))

        # Optimize the model
        history = []
        for epoch in range(1, epochs + 1):
            metrics = {
                "train.loss": [],
                "train.raw_acc": [],
                "train.mjpeg_acc": [],
                "train.adv_loss": [],
                "val.ssim": [],
                "val.psnr": [],
                "val.crop_acc": [],
                "val.scale_acc": [],
                "val.mjpeg_acc": [],
            }

            gc.collect()
            self.encoder.train()
            self.decoder.train()

            # Optimize critic-adversary
            if use_critic or use_adversary:
                iterator = tqdm(train, ncols=0)
                for frames in iterator:
                    frames, data = make_pair(frames, self.data_dim, use_bit_inverse=use_bit_inverse)
                    wm_frames = self.encoder(frames, data)
                    adv_loss = 0.0
                    if use_critic:
                        adv_loss += torch.mean(self.critic(frames) - self.critic(wm_frames))
                    if use_adversary:
                        adv_loss -= functional.binary_cross_entropy_with_logits(
                            self.decoder(self.adversary(wm_frames)), data)
                    D_opt.zero_grad()
                    adv_loss.backward()
                    D_opt.step()
                    for p in self.critic.parameters():
                        p.data.clamp_(-0.1, 0.1)

                    metrics["train.adv_loss"].append(adv_loss.item())
                    iterator.set_description("Adversary | %s" % np.mean(metrics["train.adv_loss"]))

            # Optimize encoder-decoder using critic-adversary
            if use_critic or use_adversary:
                iterator = tqdm(train, ncols=0)
                for frames in iterator:
                    frames, data = make_pair(frames, self.data_dim, use_bit_inverse=use_bit_inverse)
                    wm_frames = self.encoder(frames, data)
                    loss = 0.0
                    if use_critic:
                        critic_loss = torch.mean(self.critic(wm_frames))
                        loss += 0.1 * critic_loss
                    if use_adversary:
                        adversary_loss = functional.binary_cross_entropy_with_logits(self.decoder(self.adversary(wm_frames)), data)
                        loss += 0.1 * adversary_loss
                    G_opt.zero_grad()
                    loss.backward()
                    G_opt.step()

            # Optimize encoder-decoder
            iterator = tqdm(train, ncols=0)
            for frames in iterator:
                frames, data = make_pair(frames, self.data_dim, use_bit_inverse=use_bit_inverse)

                wm_frames = self.encoder(frames, data)
                wm_raw_data = self.decoder(noise(wm_frames))
                wm_mjpeg_data = self.decoder(mjpeg(wm_frames))

                loss = 0.0
                loss += functional.binary_cross_entropy_with_logits(wm_raw_data, data)
                loss += functional.binary_cross_entropy_with_logits(wm_mjpeg_data, data)
                G_opt.zero_grad()
                loss.backward()
                G_opt.step()

                metrics["train.loss"].append(loss.item())
                metrics["train.raw_acc"].append(get_acc(data, wm_raw_data))
                metrics["train.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))
                iterator.set_description("Epoch %s | Loss %.3f | Raw %.3f | MJPEG %.3f" % (
                    epoch,
                    np.mean(metrics["train.loss"]),
                    np.mean(metrics["train.raw_acc"]),
                    np.mean(metrics["train.mjpeg_acc"]),
                ))

            # Validate
            gc.collect()
            self.encoder.eval()
            self.decoder.eval()
            iterator = tqdm(val, ncols=0)
            with torch.no_grad():
                for frames in iterator:
                    frames = frames.cuda()
                    data = torch.zeros((frames.size(0), self.data_dim)).random_(0, 2).cuda()

                    wm_frames = self.encoder(frames, data)
                    wm_crop_data = self.decoder(mjpeg(crop(wm_frames)))   
                    wm_scale_data = self.decoder(mjpeg(scale(wm_frames))) 
                    wm_mjpeg_data = self.decoder(mjpeg(wm_frames))        

                    metrics["val.ssim"].append(ssim(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())
                    metrics["val.psnr"].append(psnr(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())
                    metrics["val.crop_acc"].append(get_acc(data, wm_crop_data))
                    metrics["val.scale_acc"].append(get_acc(data, wm_scale_data))
                    metrics["val.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))

                    iterator.set_description(
                        "Epoch %s | SSIM %.3f | PSNR %.3f | Crop %.3f | Scale %.3f | MJPEG %.3f" % (
                            epoch,
                            np.mean(metrics["val.ssim"]),
                            np.mean(metrics["val.psnr"]),
                            np.mean(metrics["val.crop_acc"]),
                            np.mean(metrics["val.scale_acc"]),
                            np.mean(metrics["val.mjpeg_acc"]),
                        )
                    )

            metrics = {k: round(np.mean(v), 3) if len(v) > 0 else "NaN" for k, v in metrics.items()}
            metrics["epoch"] = epoch
            history.append(metrics)
            pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
            with open(os.path.join(log_dir, "metrics.json"), "wt") as fout:
                fout.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))

            torch.save(self, os.path.join(log_dir, "model.pt"))
            G_scheduler.step(metrics["train.loss"])

        return history

    def save(self, path_to_model):
        torch.save(self, path_to_model)

    def load(path_to_model):
        return torch.load(path_to_model)

    def encode(self, video_in, data, video_out, fps):
        assert len(data) == self.data_dim
        measure = []
        print("Encoding Watermark to Video...")

        video_in = cv2.VideoCapture(video_in)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video Total Frame:", length)
        print("Video Width:", width)
        print("Video Height:", height)

        data = torch.FloatTensor([data]).cuda()
        video_out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # (W, H)

        for _ in tqdm(range(length)):
            start = time.time()
            _, frame = video_in.read()
            
            frame = torch.from_numpy(frame).float().div_(127.5).sub_(1.0).unsqueeze(0) # (L, H, W, 3)
            frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda() # (1, 3, L, H, W)
            wm_frame = self.encoder(frame, data)                  # (1, 3, L, H, W)
            wm_frame = torch.clamp(wm_frame, min=-1.0, max=1.0)
            wm_frame = ((wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5).detach().cpu().numpy().astype("uint8")

            video_out.write(wm_frame)
            end = time.time()
            measure.append(end-start)
        video_out.release()
        return measure
        
    def decode(self, video_in):
        video_in = cv2.VideoCapture(video_in)
        length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Extracing Data Watermark...")

        for _ in tqdm(range(length)):
            start = time.time()
            _, frame = video_in.read()

            frame = torch.from_numpy(frame).float().div_(127.5).sub_(1.0).unsqueeze(0)  # (L, H, W, 3)
            frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda() # (1, 3, L, H, W) --> decode each frame
            data = self.decoder(frame)[0].detach().cpu().numpy()
            end = time.time()
            yield data, end-start

if __name__ == "__main__":
    print("**** Inference Configurations ****")
    print(f"1. Model: RivaGAN")
    print(f"2. Device: {device}")
    print(f"3. Data Dimension: {args.data_dim}")
    print(f"4. Model Weight: {args.model_weight}")
    print(f"5. Video Location: {args.video_location}")
    print(f"6. Output FPS: {args.fps}")
    print()

    accuracy = []
    model = RivaGAN(data_dim=args.data_dim)
    model = RivaGAN.load(f"{args.model_weight}")
    print("Model's weight is loaded!")

    # Data
    if args.random_data == "Yes":
        data = tuple(random.choices([0,1], k=args.data_dim))
    else:
        data = args.your_data.replace(" ","")
        tmp = [int(i) for i in data]
        data = tuple(tmp)
    print(f"Data: {data[:4]} {data[4:8]} {data[8:12]} {data[12:16]} {data[16:20]} {data[20:24]} {data[24:28]} {data[28:32]}")
    
    # Encode
    start1 = time.time()
    measure_encoder_eachframe = model.encode(args.video_location, data, f"./inference_output{args.data_dim}.avi", args.fps)
    end1 = time.time()

    # Decode
    measure_decoder_eachframe = []
    data2 = torch.tensor(data)
    start2 = time.time()
    recovered_data = model.decode(f"./inference_output{args.data_dim}.avi")
    with open('output_log.txt', 'w') as f:
        f.write(f"Original Data:  {data[:4]} {data[4:8]} {data[8:12]} {data[12:16]} {data[16:20]} {data[20:24]} {data[24:28]} {data[28:32]}\n")
        for extracted_wm, y in recovered_data:
            accuracy.append((torch.tensor(extracted_wm)>=0.0).eq(data2>=0.5).sum().float().item()/torch.tensor(extracted_wm).numel())
            extracted_wm[extracted_wm>=0] = 1
            extracted_wm[extracted_wm<0] = 0
            extracted_wm = extracted_wm.astype(int)
            extracted_wm = tuple(extracted_wm)
            f.write(f"Extracted Data: {extracted_wm[:4]} {extracted_wm[4:8]} {extracted_wm[8:12]} {extracted_wm[12:16]} {extracted_wm[16:20]} {extracted_wm[20:24]} {extracted_wm[24:28]} {extracted_wm[28:32]}\n")
            measure_decoder_eachframe.append(y)
        end = time.time()

    # Measure Inference
    print(f"Inference Time (Complete): {(end-start1):.2f}s")
    print(f"Inference Time on Full Frame (Encoder Only): {(end1-start1):.3f}s")
    print(f"Inference Time on Each Frame (Encoder Only): {sum(measure_encoder_eachframe)/len(measure_encoder_eachframe):.3f}s")
    print(f"Inference Time on Full Frame (Decoder Only): {(end-start2):.3f}s")
    print(f"Inference Time on Each Frame (Decoder Only): {sum(measure_decoder_eachframe)/len(measure_decoder_eachframe):.3f}s")

    # Measure Accuracy
    print(f"Average Accuracy Data: {sum(accuracy)/len(accuracy):.2f}")