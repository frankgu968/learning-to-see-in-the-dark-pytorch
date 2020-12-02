FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

COPY ./ ./

RUN apt-get update
RUN apt-get install -y vim libgl1-mesa-glx git
RUN pip install -r requirements.txt
RUN conda install -c conda-forge opencv

# To install nvidia apex
RUN git clone https://github.com/NVIDIA/apex /usr/local/src/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /usr/local/src/apex

