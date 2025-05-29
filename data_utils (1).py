import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset
import numpy as np

# cubes is a list of our 3d images
class CubeDataset(torch.utils.data.Dataset): # returns dataset on cpu
  def __init__(self, cubes):
    self.cubes = cubes
    np.random.seed(42)

  def __len__(self):
    return len(self.cubes)

  def __getitem__(self, idx):
    return self.cubes[idx]