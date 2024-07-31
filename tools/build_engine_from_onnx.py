import os
import json
import tensorrt as trt

def setDynamicRange(network, json_file: str):
    """Sets ranges for network layers."""
    with open(json_file) as file:
        quant_param_json = json.load(file)
    act_quant = quant_param_json["act_quant_info"]

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if act_quant.__contains__(input_tensor.name):
            value = act_quant[input_tensor.name]
            tensor_max = abs(value)
            tensor_min = -abs(value)
            input_tensor.dynamic_range = (tensor_min, tensor_max)

    for i in range(network.num_layers):
        layer = network.get_layer(i)

        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)

            if act_quant.__contains__(tensor.name):
                value = act_quant[tensor.name]
                tensor_max = abs(value)
                tensor_min = -abs(value)
                tensor.dynamic_range = (tensor_min, tensor_max)

def build_engine(
    onnx_file: str, engine_file: str,
    fp16: bool = True, int8: bool = False, 
    int8_scale_file: str = None,
    explicit_batch: bool = True, 
    dynamic_shapes: map = {},
    dynamic_batch_size:int = 1,
    workspace: int = 4294967296<<4, # 4GB
    ):
    TRT_LOGGER = trt.Logger()
    """
    Build a TensorRT Engine with given onnx model.

    Flag int8, fp16 specifies the precision of layer:
        For building FP32 engine: set int8 = False, fp16 = False, int8_scale_file = None
        For building FP16 engine: set int8 = False, fp16 = True, int8_scale_file = None
        For building INT8 engine: set int8 = True, fp16 = True, int8_scale_file = 'json file name'

    """

    if int8 is True:
        if int8_scale_file is None:
            raise ValueError('Build Quantized TensorRT Engine Requires a JSON file which specifies variable scales, '
                             'however int8_scale_file is None now.')

    builder = trt.Builder(TRT_LOGGER)
    if explicit_batch:
        network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    else: network = builder.create_network()
    
    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, TRT_LOGGER)
    # config.max_workspace_size = workspace

    # builder.max_batch_size = 

    if len(dynamic_shapes) > 0:
        print(f"===> using dynamic shapes: {str(dynamic_shapes)}")
        builder.max_batch_size = dynamic_batch_size
        profile = builder.create_optimization_profile()

        for binding_name, dynamic_shape in dynamic_shapes.items():
            min_shape, opt_shape, max_shape = dynamic_shape
            profile.set_shape(
                binding_name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f'ONNX file {onnx_file} not found')

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    if fp16: config.set_flag(trt.BuilderFlag.FP16)
    if int8_scale_file is not None and int8:
        config.set_flag(trt.BuilderFlag.INT8)
        setDynamicRange(network, int8_scale_file)
    
    engine = builder.build_serialized_network(network, config)

    with open(engine_file, "wb") as f:
        f.write(engine.serialize())


if __name__ == "__main__":
        build_engine(onnx_file='rgbt_ca_rtdetrv2_589_m3fd_op18.onnx', 
                 int8_scale_file='rgbt_ca_rtdetrv2_589_m3fd_int8_SYMM_LINEAR_PERCHANNEL_dynamic_quantized.json', 
                 engine_file='rgbt_ca_rtdetrv2_589_m3fd_int8_SYMM_LINEAR_PERCHANNEL.engine', 
                #  dynamic_shapes={'b' : [(2,3,640,640),(16,3,640,640),(32,3,640,640)]},
                 int8=True)

