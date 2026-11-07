import onnx
from onnx import helper

model = onnx.load("../../C06/output/mobilenet_v2.onnx")

count = 0

for node in model.graph.node:
    if node.op_type != "Conv":
        continue

    count = count + 1
    group = 1

    for attr in node.attribute:
        if attr.name == "group":
            group = helper.get_attribute_value(attr)


    print(f"conv {count:2d} group = {group}")
