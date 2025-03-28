#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# New VM
rm -rf yolov3 weights coco
git clone https://github.com/ultralytics/yolov3
bash yolov3/weights/download_yolov3_weights.sh && cp -r weights yolov3
bash yolov3/data/get_coco_dataset.sh
git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make && cd ../.. && cp -r cocoapi/PythonAPI/pycocotools yolov3
sudo shutdown

# Re-clone
rm -rf mnist                                   # Warning: remove existing
git clone https://github.com/ultralytics/mnist # master
# git clone -b test --depth 1 https://github.com/ultralytics/mnist test  # branch
#cp -r weights mnist && cd mnist
