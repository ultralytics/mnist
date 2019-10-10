import torch
import cv2
import numpy as np
from tqdm import tqdm
import glob
from utils import torch_utils
import pretrainedmodels

device = torch_utils.select_device()

# Load model
model_name = 'resnet101'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

# adjust last layer
n = 2  # desired classes
filters = model.last_linear.weight.shape[1]
model.last_linear.bias = torch.nn.Parameter(torch.zeros(n))
model.last_linear.weight = torch.nn.Parameter(torch.zeros(n, filters))
model.last_linear.out_features = n

chkpt = torch.load('resnet101.pt', map_location=device)
model.load_state_dict(chkpt['model'], strict=True)

dir = './samples'
results = []
model.eval()
with torch.no_grad():
    for file in tqdm(glob.glob('%s/*.*' % dir)[:9000]):
        img = cv2.resize(cv2.imread(file), (128, 128))  # BGR
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.expand_dims(img, axis=0)  # add batch dim
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.Tensor(img)

        results.append(model(img))  # output

print(results)
