import onnx
import onnx_graphsurgeon as gs
import numpy as np

model = onnx.load("rgbt_yolov5_m3fd_op13.onnx")
graph = gs.import_onnx(model)
# graph.fold_constants().cleanup()

input0 = graph.inputs[0]
input1 = graph.inputs[1]

new_input = gs.Variable(name="input", dtype=np.float32, shape=(2, ) + tuple(input0.shape[1:]))

# slice1 = gs.Node(
#     op="Slice",
#     inputs=[new_input],
#     outputs=[gs.Variable(name="slice1_output", dtype=np.float32)],
#     attrs={"axes": [0], "starts": [0], "ends": [1]}
# )
# slice2 = gs.Node(
#     op="Slice",
#     inputs=[new_input],
#     outputs=[gs.Variable(name="slice2_output", dtype=np.float32)],
#     attrs={"axes": [0], "starts": [1], "ends": [2]}
# )

split_node = gs.Node(
    op="Split",
    inputs=[new_input],
    outputs=[
        gs.Variable(name="split_output1", dtype=np.float32, shape=(1, ) + tuple(input0.shape[1:])),
        gs.Variable(name="split_output2", dtype=np.float32, shape=(1, ) + tuple(input1.shape[1:]))
    ],
    attrs={"axis": 0}
)

conv0 = [node for node in graph.nodes if node.name == "/rgb_conv_1/conv/Conv"][0]
conv1 = [node for node in graph.nodes if node.name == "/t_conv_1/conv/Conv"][0]

conv0.inputs[0] = split_node.outputs[0]
conv1.inputs[0] = split_node.outputs[1]

graph.inputs = [new_input]
graph.nodes.append(split_node)

graph.cleanup().toposort()
new_model = gs.export_onnx(graph)
onnx.save(new_model, "rgbt_yolov5_m3fd_op13_one_input.onnx")
