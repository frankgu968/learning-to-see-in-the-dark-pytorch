# learning-to-see-in-the-dark-pytorch
PyTorch Implementation of the Learning To See In the Dark paper

## Inference on Pretrained Model
After the model has been trained inference can be run with the following command.

`$ python infer.py RAW_IMAGE_PATH OUTPUT_IMAGE EXPOSURE_RATIO`

Input must be a raw image in the **.ARW** format and output must be a **.PNG** format. Exposure is the ratio between the exposure time of the input photo and the desired output. Best performance occurs on dark images with ratio: [100, 250]

## Using Tensorboard
Run `tensorboard --logdir=runs` to start Tensorboard.  
Ensure that auto-reload is enabled on Tensorboard to fetch continuous updates as the model trains.  
If you are running different workloads/experiments, remember to change the `run_name` configuration!