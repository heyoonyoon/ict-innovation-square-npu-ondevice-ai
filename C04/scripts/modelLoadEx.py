import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights



weights = MobileNet_V2_Weights.IMAGENET1K_V2
print(weights)

model = mobilenet_v2(weights=weights)
model.eval()

print(model.__class__.__name__)

device = torch.device("cpu")
print(f"사용디바이스: {device}")
print(f"CUDA 사용가능여부: {torch.cuda.is_available()}")
