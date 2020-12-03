from torch.utils.data import Dataset
import numpy as np
import glob
import os
import rawpy
from pathlib import Path
import cv2

def pack_raw(raw):
  # convert to 3 channel
  print(raw)
  print("RawPy Process")
  im = raw.postprocess(half_size=True, output_bps=16)
  print("Filter Black")
  im = np.maximum(im - 2047.0, 0) / (65535.0 - 2047.0)  # subtract the black level
  print("Cast to Float32")
  im = im.astype(np.float32)

  return im

class LTSIDDataset(Dataset):
  def __init__(self, input_dir, truth_dir, preprocess, preprocess_dir, collection='train', transforms=None):
    self.input_dir = input_dir
    self.truth_dir = truth_dir
    self.preprocess = preprocess
    self.preprocess_dir = preprocess_dir
    self.collection = collection
    self.transforms = transforms
    
    # Load different data collections for train, test, validation split
    self.fn_prefix = '0'
    if collection == 'train':
      self.fn_prefix = '0'
    elif collection == 'validation':
      self.fn_prefix = '1'
    elif collection == 'test':
      self.fn_prefix = '2'
    else:
      print('Unsupported dataset collection: {}. Must be in [train, validation, test]'.format(collection))
      exit(1)
    print('Loading dataset collection: {}'.format(collection))
    self.input_fns = glob.glob(input_dir + self.fn_prefix + '*_00*.ARW')  # All the input image filenames
    self.truth_fns = glob.glob(truth_dir + self.fn_prefix + '*.ARW')  # All the ground truth filenames
    
    # UNCOMMENT FOR SMALLER DEBUG DATASET
    #self.truth_fns = self.truth_fns[0:int(len(self.truth_fns)/8)]
    
    # Pre-allocate lists for images
    self.truth_images = [None] * len(self.truth_fns)
    self.input_images = [None] * len(self.input_fns) 
    self.input_truth_map = [None] * len(self.input_fns) # Array mapping a training image index to the corresponding truth index

    # Load images
    self.preprocess_file = self.preprocess_dir + collection + '.npy'
    if os.path.exists(self.preprocess_file) & ~self.preprocess:
      # Load existing preprocessed data
      print('Preprocessed image file found, loading...')
      self.load_preprocessed()
    else:
      print('Preprocessing images...')
      Path(self.preprocess_dir).mkdir(exist_ok=True)
      self.preprocess_images()
      self.save_preprocessed()

  def load_preprocessed(self):
    load_arr = np.load(self.preprocess_file, allow_pickle=True)
    self.truth_images = load_arr[0]
    self.input_images = load_arr[1]
    self.input_truth_map = load_arr[2]
    print('Preprocessed data loaded!')

  def save_preprocessed(self):
    save_arr = np.array([self.truth_images, self.input_images, self.input_truth_map], dtype=object)
    np.save(self.preprocess_file, save_arr)
    print('Preprocessed data saved!')

  def preprocess_images(self):
    input_cnt = 0 # Index counter for the training images
    for idx, truth_fn in enumerate(self.truth_fns):
      print('Preprocessing {} / {} {} IDs'.format(idx+1, len(self.truth_fns), self.collection))
      input_id = int(os.path.basename(truth_fn)[0:5])
      truth_fn = os.path.basename(truth_fn)
      input_files = glob.glob(self.input_dir + '%05d_00*.ARW' % input_id) # Multiple training files
      truth_exposure = float(truth_fn[9:-5])    # Get the exposure time from the ground truth filename

      # Load the ground truth image
      truth_raw = rawpy.imread(self.truth_dir + truth_fn)
      im = truth_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
      self.truth_images[idx] = (im/65535.0).astype(np.float32)

      # Load the associated training images
      for input_fn in input_files:
        input_fn = os.path.basename(input_fn)
        input_exposure = float(input_fn[9:-5])
        ratio = min(truth_exposure / input_exposure, 300)     # Calculate exposure ratio for simple scaling
        raw = rawpy.imread(self.input_dir + input_fn)         # Load image
        self.input_images[input_cnt] = pack_raw(raw) * ratio  # Scale the pixel values by the exposure time ratio
        self.input_truth_map[input_cnt] = idx                 # Set the index of the corresponding ground truth image
        input_cnt = input_cnt + 1

    # Removing extra allocated space
    self.input_images = self.input_images[0:input_cnt]
    self.input_truth_map = self.input_truth_map[0:input_cnt]


  def __len__(self):
    return len(self.input_images)

  def __getitem__(self, idx):
    truth_idx = self.input_truth_map[idx]
    sample = {
      'train': self.input_images[idx],
      'truth': self.truth_images[truth_idx]
    }

    if self.transforms:
      sample = self.transforms(sample)

    return sample