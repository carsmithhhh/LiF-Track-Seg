import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset
import numpy as np
from data_utils import *
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, patch_model, pos_model, encoder, decoder):
        self.patch_model = patch_model
        self.pos_model = pos_model
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, norm_x):
        patch_embeddings = self.patch_model(norm_x)
        self.pos_model = 
        #def forward(self, pos: torch.Tensor) -> torch.Tensor:

def pretrain(model, train_loader, optimizer, epoch, epochs, device=device, patch_size):
    