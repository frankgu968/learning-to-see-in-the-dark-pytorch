import numpy as np
import torch
from matplotlib import pyplot as plt

# Need to write custom transforms since the ground truth and the training images are different!
class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""

  def __call__(self, sample):
    train, truth = sample['train'], sample['truth']
    train = torch.from_numpy(train).transpose(0, 2)  # CHW
    truth = torch.from_numpy(truth).transpose(0, 2)
    return {
      'train': train,
      'truth': truth,
    }

class RandomTranspose(object):
  """ Randomly transpose H and W """
  def __init__(self, p=0.5):
    assert p >= 0 and p <= 1
    self.probability = p

  def __call__(self, sample):
    train, truth = sample['train'], sample['truth']
    if np.random.rand() >= self.probability:
      train = np.transpose(train, (0, 2, 1))
      truth = np.transpose(truth, (0, 2, 1))

    return {
      'train': train, 
      'truth': truth
    }

class RandomHorizontalFlip(object):
  """ Randomly flip horizontally """
  def __init__(self, p=0.5):
    assert p >= 0 and p <= 1
    self.probability = p

  def __call__(self, sample):
    train, truth = sample['train'], sample['truth']
    if np.random.rand() >= self.probability:
      train = torch.fliplr(train)
      truth = torch.fliplr(truth)

    return {
      'train': train, 
      'truth': truth
    }

class RandomVerticalFlip(object):
  """ Randomly flip vertically """
  def __init__(self, p=0.5):
    assert p >= 0 and p <= 1
    self.probability = p

  def __call__(self, sample):
    train, truth = sample['train'], sample['truth']
    if np.random.rand() >= self.probability:
      train = torch.flipud(train)
      truth = torch.flipud(truth)

    return {
      'train': train, 
      'truth': truth
    }

class RandomCrop(object):
  """ Randomly transpose H and W """
  def __init__(self, ps=0.5):
    assert ps >= 64 and ps <= 1024
    self.patch_size = ps

  def __call__(self, sample):
    train, truth = sample['train'], sample['truth']
    # crop
    H = train.shape[0]
    W = train.shape[1]

    ps = self.patch_size
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)

    train = train[yy:yy + ps, xx:xx + ps, :]
    truth = truth[yy:yy + ps, xx:xx + ps, :]

    return {
      'train': train, 
      'truth': truth
    }