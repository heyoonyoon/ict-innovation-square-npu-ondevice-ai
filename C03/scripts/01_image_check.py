from PIL import Image
from pathlib import Path

image_path = Path("../images/red_square.jpg")


print(f'image_path: {image_path}')
print(f'파일 존재 여부: {image_path.exists()}')

image = Image.open(image_path)
print(f'이미지 형식:{image.format}')
print(f'이미지 모드:{image.mode}')
print(f'이미지 크기:{image.size}')

image2 = Image.open(image_path).convert('RGB')

import numpy as np

array = np.array(image2)
print("Numpy shape",array.shape)
print("Numpy dtype",array.dtype)
print(f'최대값: {array.max()}, 최소값: {array.min()}')

resized = image2.resize((224, 224))
print("resized size:", resized.size)
resized.save('../images/resized.jpg')