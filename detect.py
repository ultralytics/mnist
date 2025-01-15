# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import torch_utils

device = torch_utils.select_device()

# Load model
model = torch_utils.load_classifier(name="resnet101", n=2)

# Load state_dict
chkpt = torch.load("resnet101.pt", map_location=device)
model.load_state_dict(chkpt["model"], strict=True)
model.eval()

dir = "./samples"
results = []
with torch.no_grad():
    for file in tqdm(glob.glob(f"{dir}/*.*")[:9000]):
        img = cv2.resize(cv2.imread(file), (128, 128))  # BGR
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.expand_dims(img, axis=0)  # add batch dim
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.Tensor(img)

        results.append(model(img))  # output

print(results)
