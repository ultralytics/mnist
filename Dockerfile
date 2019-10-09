# Start from Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:19.08-py3

# Install dependencies (pip or conda)
RUN pip install -U gsutil
# RUN pip install -U -r requirements.txt
# RUN conda update -n base -c defaults conda
# RUN conda install -y -c anaconda future numpy opencv matplotlib tqdm pillow
# RUN conda install -y -c conda-forge scikit-image tensorboard pycocotools
RUN pip install pretrainedmodels

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build
# rm -rf yolov3  # Warning: remove existing
# git clone https://github.com/ultralytics/yolov3 && cd yolov3 && python3 detect.py
# sudo docker image prune -af && sudo docker build -t ultralytics/yolov3:v0 .

# Run
# sudo nvidia-docker run --ipc=host ultralytics/yolov3:v0 python3 detect.py

# Run with local directory access
# sudo nvidia-docker run --ipc=host --mount type=bind,source="$(pwd)"/knife_classifier,target=/usr/src/knife_classifier ultralytics/mnist:v0 python3 train_resnet.py

# Pull and Run with local directory access
# export tag=ultralytics/mnist:v0 && sudo docker pull $tag && sudo nvidia-docker run --ipc=host --mount type=bind,source="$(pwd)"/knife_classifier,target=/usr/src/knife_classifier $tag python3 train_resnet.py

# Build and Push
# export tag=ultralytics/mnist:v0 && sudo docker build -t $tag . && docker push $tag

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Run bash for loop
# sudo nvidia-docker run --ipc=host ultralytics/yolov3:v0 while true; do python3 train.py --evolve; done
