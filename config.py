class Config:
  def __init__(self, cfg):
    self.input_dir = cfg['input_dir']
    self.truth_dir = cfg['truth_dir']
    self.checkpoint_dir = cfg['checkpoint_dir']
    self.output_dir = cfg['output_dir']
    self.preprocess = bool(cfg['preprocess'])
    self.preprocess_dir = cfg['preprocess_dir']
    self.checkpoint_path = self.checkpoint_dir + 'latest.t7'
    self.checkpoint_to_load = self.checkpoint_dir + str(cfg['checkpoint_to_load'])
    self.patch_size = int(cfg['patch_size'])
    self.save_interval = int(cfg['save_interval'])   # epochs
    self.batch_size = int(cfg['batch_size'])
    self.initial_learning_rate = float(cfg['initial_learning_rate'])
    self.epochs = int(cfg['epochs'])
    self.run_name = cfg['run_name']
    
    # Multi GPU
    self.num_nodes = 1
    self.gpus_per_node = 1

    if ('num_nodes' in cfg.keys()):
        self.num_nodes = cfg['num_nodes']
    if ('gpus_per_node' in cfg.keys()):
        self.gpus_per_node = cfg['gpus_per_node']
