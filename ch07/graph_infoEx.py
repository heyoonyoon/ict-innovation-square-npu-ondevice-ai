import onnx

model = onnx.load('../../C06/output/mobilenet_v2.onnx')


def get_shape(value_info):
    dims = []

    tensor_type = value_info.type.tensor_type

    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(dim.dim_value)
        else:
            dims.append("?")

    return dims

for item in model.graph.input:
    #print(item)
    print('Name:', item.name)

    print("shape:", get_shape(item))

print()
for item in model.graph.output:
    #print(item)
    print('Name:', item.name)

    print("shape:", get_shape(item))

print()
nodes = model.graph.node
for i, node in enumerate(nodes[:20], start=1):
    print(f"[{i}]")
    print(f"name: {node.name}")
    print(f"operator: {node.op_type}")
    print(f"inputs: {list(node.input)}")
    print(f"outputs: {list(node.output)}")
    print('-'* 100)
    print()

initializers = model.graph.initializer
print(len(initializers))

print("== initializer ==")
for i, tensor in enumerate(initializers[:20], start=1):
    print(f"[{i}]")
    print(f"dim: {list(tensor.dims)}")
    print(f"data_type: {tensor.data_type}")
    print('-'* 100)
    print()


conv_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
print("Conv length: ",len(conv_nodes))

from onnx import helper

for i, node in enumerate(conv_nodes[:5], start=1):
    print(f"[conv {i}]")
    print(f"name: {node.name}")
    print(f"inputs: {list(node.input)}")
    print(f"outputs: {list(node.output)}")
    print('attributes:')

    for attr in node.attribute:
        value = helper.get_attribute_value(attr)
        print(f"  {attr.name}: {value}")
        print('-'* 100)
        print()











