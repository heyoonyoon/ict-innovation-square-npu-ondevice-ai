from pathlib import Path
import time
import torch
import onnx
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.quantization import mobilenet_v2


def export_to_onnx(model, input_shape, output_file):
    output_file  = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(
        *input_shape,
        dtype=torch.float32
    )

    start = time.perf_counter()
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_file),
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        external_data=False
    )
    end = time.perf_counter()
    export_time = end - start
    print(f"변환시간: {export_time} seconds")

    return output_file

def check_onnx_model(onnx_file):
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    file_size_mb = Path(onnx_file).stat().st_size / 1024 / 1024

    print("=== ONNX 확인 ===")
    print("Checker : 정상")
    print(f"파일크기: {file_size_mb:.2f}:MB")

    print("Input")
    for input_info in model.graph.input:
        print("-", input_info.name)
    print("Output")
    for output_info in model.graph.output:
        print("-", output_info.name)


weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights = weights)
model.eval()

test_input = torch.randn(
    1,3,224,224
)

with torch.inference_mode():
    pytorch_output = model(test_input)

print(f"pytorch 출력 shape: {pytorch_output.shape}")

onnx_file = export_to_onnx(
    model,
    input_shape=(1,3,224,224),
    output_file="../output/mobilenet_v2.onnx"
)

check_onnx_model(onnx_file)