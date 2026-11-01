from PIL import Image
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


device = torch.device('cpu')
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).to(device)

model.eval()

image = Image.open('../C04_data/images/cat.jpg').convert('RGB')

input_batch = (
    weights.transforms()(image)
    .unsqueeze_(0)
    .to(device)
)

import time

start = time.perf_counter()
with torch.no_grad():
    output = model(input_batch)


end = time.perf_counter()


elapsed= end - start
print("Inference time: ", elapsed, "seconds")
print("Inference time:", elapsed * 1000, "ms")
