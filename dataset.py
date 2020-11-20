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
  def __init__(self, train_dir, truth_dir, train=True, transforms=None):
    self.train_dir = train_dir
    self.truth_dir = truth_dir
    self.train_fns = glob.glob(train_dir + '0*_00*.ARW')  # All the training filenames
    self.truth_fns = glob.glob(truth_dir + '0*.ARW')  # All the ground truth filenames
    # Pre-allocate lists for images
    self.truth_images = [None] * len(self.truth_fns)
    self.train_images = [None] * len(self.train_fns)
    self.train_truth_map = [None] * len(self.train_fns) # Array mapping a training image index to the corresponding truth index
    # Set the keys for different exposure levels
    self.load_images()
    self.train = train
    self.transforms = transforms

  def load_images(self):
    train_cnt = 0 # Index counter for the training images
    for idx, truth_fn in enumerate(self.truth_fns):
      print('Loading {} / {} training IDs'.format(idx, len(self.truth_fns)))
      train_id = int(os.path.basename(truth_fn)[0:5])
      truth_fn = os.path.basename(truth_fn)
      train_files = glob.glob(self.train_dir + '%05d_00*.ARW' % train_id)     # Multiple training files
      truth_exposure = float(truth_fn[9:-5])    # Get the exposure time from the ground truth filename

      # Load the ground truth image
      truth_raw = rawpy.imread(self.truth_dir + truth_fn)
      im = truth_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
      self.truth_images[idx] = np.float32(im / 65535.0)

      # Load the associated training images
      for train_fn in train_files:
        train_fn = os.path.basename(train_fn)
        train_exposure = float(train_fn[9:-5])
        ratio = min(truth_exposure / train_exposure, 300)     # Calculate exposure ratio for simple scaling
        raw = rawpy.imread(self.train_dir + train_fn)         # Load image
        self.train_images[train_cnt] = pack_raw(raw) * ratio  # Scale the pixel values by the exposure time ratio
        self.train_truth_map[train_cnt] = idx                 # Set the index of the corresponding ground truth image
        train_cnt = train_cnt + 1

  def __len__(self):
    return len(self.train_images)

  def __getitem__(self, idx):
    truth_idx = self.train_truth_map[idx]
    sample = {
      'train': self.train_images[idx],
      'truth': self.truth_images[truth_idx]
    }

    if self.transforms:
      sample = self.transforms(sample)

    return sample
