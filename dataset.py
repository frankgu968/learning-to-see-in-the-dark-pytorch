from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import os
import rawpy


def pack_raw(raw):
  # pack Bayer image to 4 channels
  im = raw.raw_image_visible.astype(np.float32)
  im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

  im = np.expand_dims(im, axis=2)
  img_shape = im.shape
  H = img_shape[0]
  W = img_shape[1]

  out = np.concatenate((im[0:H:2, 0:W:2, :],
                        im[0:H:2, 1:W:2, :],
                        im[1:H:2, 1:W:2, :],
                        im[1:H:2, 0:W:2, :]), axis=2)
  return out


class LTSIDDataset(Dataset):
  def __init__(self, train_dir, truth_dir, transforms=None):
    self.train_dir = train_dir
    self.truth_dir = truth_dir
    self.train_fns = glob.glob(train_dir + '0*.ARW')  # All the training filenames
    # Create container for images
    self.truth_images = [None] * len(self.train_fns)
    self.train_images = [None] * len(self.train_fns)
    # Set the keys for different exposure levels
    self.load_images()
    self.transforms = transforms

  def load_images(self):
    for idx, train_fn in enumerate(self.train_fns):
      train_id = int(os.path.basename(train_fn)[0:5])
      truth_file = glob.glob(self.truth_dir + '%05d_00*.ARW' % train_id)[0]   # 1 ground truth file
      truth_fn = os.path.basename(truth_file)
      truth_exposure = float(truth_fn[9:-5])
      # Load ground-truth image
      truth_raw = rawpy.imread(truth_file)
      im = truth_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
      self.truth_images[idx] = np.float32(im / 65535.0)

      train_fn = os.path.basename(train_fn)
      train_exposure = float(train_fn[9:-5])
      ratio = min(truth_exposure / train_exposure, 300)  # Calculate exposure ratio for simple scaling
      # Load image
      raw = rawpy.imread(self.train_dir + train_fn)
      self.train_images[idx] = pack_raw(raw) * ratio

  def __len__(self):
    return len(self.train_images)

  def __getitem__(self, idx):
    sample = {
      'train': self.train_images[idx],
      'truth': self.truth_images[idx]
    }

    if self.transforms:
      sample = self.transforms(sample)

    return sample

class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""

  def __call__(self, sample):
    train, truth = sample['train'], sample['truth']

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    # train = train.transpose((2, 0, 1))
    # truth = truth.transpose((2, 0, 1))
    return {'train': torch.from_numpy(train),
            'truth': torch.from_numpy(truth),
            }
