FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY ./ ./

RUN pip install -r requirements.txt
RUN conda install -c anaconda opencv

