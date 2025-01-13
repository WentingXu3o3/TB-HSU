FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
#ubuntu18.04

# install required packages
RUN apt-get update && apt-get install -y \
    git \
    libsparsehash-dev \
    libgl1-mesa-glx

# install torchsparse
RUN pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# Copy the environment file and create the conda environment
COPY environment.yaml ./
RUN conda env create -n dsg_3d -f environment.yaml

# Ensure the conda environment is activated when a new shell is started
SHELL ["/bin/bash", "-c"]
# RUN echo "source activate dsg_3d" >> ~/.bashrc
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate dsg_3d" >> ~/.bashrc


 
CMD ["/bin/bash"] 
