import onnx

MODEL_PATH = "../output/mobilenet_v2.onnx"

model = onnx.load(MODEL_PATH)

print("IR version: ", model.ir_version)
print("Producer: ", model.producer_name)


print("ONNX Moder 검사시작")

onnx.checker.check_model(model)

print("ONNX Model 검사 성공")

graph = model.graph


print("== input ==")

for item in graph.input:
    print(f"name : {item.name}")

    tensor_type= item.type.tensor_type

    shape = []

    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")

print(f"shape: {shape}")




print("== output ==")

for item in graph.output:
    print(f"name : {item.name}")

    tensor_type= item.type.tensor_type

    shape = []

    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")

print(f"shape: {shape}")



print()
print("Graph name: ", graph.name)
print("Node count: ", len(graph.node))
print("Initializer count: ", len(graph.initializer))

