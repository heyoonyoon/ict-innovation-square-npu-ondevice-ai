from pathlib import Path
import onnx
import pandas as pd

MODEL_FILE = Path("../../C06/output/mobilenet_v2.onnx")

OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = onnx.load(MODEL_FILE)
onnx.checker.check_model(model)

graph = model.graph

print("onnx model: ", MODEL_FILE)
print("전체 Node : ", len(graph.node))

initializer_dict = {initializer.name: initializer for initializer in graph.initializer}


def get_attribute(node, attr_name, default=None):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)

    return default


conv_rows = []

for index, node in enumerate(graph.node):
    if node.op_type != "Conv":
        continue

    group = get_attribute(node, "group", 1)
    strides = get_attribute(node, "strides", [])
    pads = get_attribute(node, "pads", [])
    kernel_shape = get_attribute(node, "kernel_shape", [])

    weight_name = (node.input[1] if len(node.input) > 2 else None)

    if weight_name in initializer_dict:
        weight_tensor = initializer_dict[weight_name]
        weight_shape = weight_tensor.dims


    # Conv 유형 분류

    if group == 1:
        conv_type = "Standard / Pointwise Conv"
    else:
        conv_type = "Grouped / Depthwise Conv"

    conv_rows.append({
        "node_index": index,
        "node_name": node.name,
        "conv_type": conv_type,
        "group": group,
        "kernel_shape": kernel_shape,
        "strides": strides,
        "pads": pads,
        "weight_shape": weight_shape,
        "input": node.input[0],
        "output": node.output[0],
    })

conv_df = pd.DataFrame(conv_rows)

print()
print("===== Conv Node 분석 =====")
print(conv_df[
          [
              "node_index",
              "conv_type",
              "group",
              "kernel_shape",
              "weight_shape",
          ]].to_string())

conv_summary = conv_df.groupby("conv_type").size().reset_index(name="count")
print()
print("===== Conv 구조 요약 =====")
print(conv_summary.to_string(index=False))

kernel_summary = conv_df.groupby(conv_df["kernel_shape"].astype(str)).size().reset_index(name="count")

kernel_summary.columns = ["kernel_shape", "count"]
print()
print("===== kernel 구조 =====")
print(kernel_summary.to_string(index=False))

depthwise_df = conv_df[conv_df["group"] > 1].copy()

print()
print("===== Group > 1 Conv =====")
print(depthwise_df[
    [
        "node_index",
        "group",
        "kernel_shape",
        "weight_shape",
    ]
      ].head(20).to_string(index=False))

conv_df.to_csv(OUTPUT_DIR / "conv_analysis.csv", index=False)
conv_summary.to_csv(OUTPUT_DIR / "conv_summary.csv", index=False)








