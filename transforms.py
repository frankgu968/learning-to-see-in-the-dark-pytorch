import numpy as np


class RandomTranspose(object):
  """ Randomly transpose H and W """
  def __init__(self, p=0.5):
    assert p >= 0 and p <= 1
    self.probability = p

  def __call__(self, sample):
    if np.random.rand() >= self.probability:
      sample = np.transpose(sample, (0, 2, 1))

    return sample