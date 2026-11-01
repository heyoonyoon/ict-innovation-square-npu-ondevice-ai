from PIL import Image
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time
from pathlib import Path
import csv


WARMUP = 10
ITERATIONS = 100
latencyies = []
OUTPUT_FILE = Path("../output/c05_latency_100.csv")

device = torch.device('cpu')
weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()
model = mobilenet_v2(weights=weights).to(device)
print(preprocess)
model.eval()


image = Image.open("../../C04/C04_data/images/cat.jpg").convert('RGB')

input_batch = (
    preprocess(image)
    .unsqueeze_(0)
    .to(device)
)


