import time
import onnxruntime as ort
import cv2 as cv
import numpy as np
import numpy
import random
from pathlib import Path

from threading import Thread
import numpy as np
import tensorrt as trt
import torchvision.models
from tqdm import tqdm

import trt_infer
from ppq import *
from ppq.api import *
from PIL import Image


BATCHSIZE        = 1
CFG_VALID_RESULT = False
EPSILON          = 1e-5

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < EPSILON)
    else:
        return np.all( a == b )

def infer_trt_once(model_path: str, sample: List[np.ndarray]) -> np.ndarray:
    """ Run a tensorrt model with given samples
    """
    logger = trt.Logger(trt.Logger.ERROR)
    with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []
    if CFG_VALID_RESULT:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
            # for sample in tqdm(samples, desc='TensorRT is running...'):
            inputs[0].host = convert_any_to_numpy(sample[0])
            inputs[1].host = convert_any_to_numpy(sample[1])

            [output] = trt_infer.do_inference(
                context, bindings=bindings, inputs=inputs, 
                outputs=outputs, stream=stream, batch_size=1)[0]
            results.append(convert_any_to_torch_tensor(output).reshape([-1, 1000]))
    else:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
            inputs[0].host = convert_any_to_numpy(sample[0])
            inputs[1].host = convert_any_to_numpy(sample[1])

            # for sample in tqdm(samples, desc='TensorRT is running...'):
            results = trt_infer.do_inference(
                context, bindings=bindings, inputs=inputs, 
                outputs=outputs, stream=stream, batch_size=1)
    return results

def infer_onnx_once(model_path: str, sample: List[np.ndarray]) -> np.ndarray:    
    onnx_infer = ort.InferenceSession(model_path)
    feed_dict, output_names = {}, []
    for input_meta in onnx_infer.get_inputs():
        name, dtype, shape = input_meta.name, input_meta.type, input_meta.shape
        
        for element in shape:
            if element is None or type(element) == str:
                raise TypeError('Dynamic input is not supported by this function.')

        if dtype == 'tensor(float)':
            feed_dict[name] = np.random.random(size=shape).astype(np.float32)
        else:
            raise Exception(f'Input {name} has unexpected data type.')
    
    for output_meta in onnx_infer.get_outputs():
        output_names.append(output_meta.name)
    onnx_infer.set_providers(['CPUExecutionProvider'])
    results = onnx_infer.run(None, {'input0':sample[0],'input1':sample[1]})
    return results

def load_image(image_path:str, input_shape:List[int]):
    h, w= input_shape[2],input_shape[3]
    im = cv.imread(str(image_path))
    im = cv.resize(im, (h, w), interpolation=cv.INTER_AREA)
    return im



directory = '/home/zhengyuhang/datasets/LLVIP_yolo/images/val'
path = Path(directory)
rgb_images = sorted(list(path.glob('*_rgb.jpg')))
t_images = sorted(list(path.glob('*_t.jpg')))
assert len(rgb_images) == len(t_images), f'The number of RGB images is not equal to the number of infrared images, {len(rgb_images)}!={len(t_images)}'
pair_list = list(zip(rgb_images, t_images))
random_list = random.sample(pair_list, 1)
# rgb_sample_list, t_sample_list = zip(*random_list)
rgb_batches, rgb_batch, t_batches, t_batch = [], [], [], []

for rgb_image, t_image in random_list:
    rgb_tensor = load_image(str(rgb_image), input_shape=[BATCHSIZE, 3, 1280, 1280])
    t_tensor = load_image(str(t_image), input_shape=[BATCHSIZE, 3, 1280, 1280])
    rgb_tensor = np.expand_dims(rgb_tensor, axis=0).transpose(0,3,1,2)
    t_tensor = np.expand_dims(t_tensor, axis=0).transpose(0,3,1,2)
    rgb_tensor = rgb_tensor.astype(np.float32)/ 255.0
    t_tensor = t_tensor.astype(np.float32)/ 255.0
    rgb_tensor = np.ascontiguousarray(rgb_tensor)
    t_tensor = np.ascontiguousarray(t_tensor)
    rgb_batch.append(rgb_tensor)
    t_batch.append(t_tensor)


trt_result = infer_trt_once(model_path='/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/tools/rgbt_yolov5_op13_quantized.engine', sample=[rgb_batch[0],t_batch[0]])
onnx_result0 = infer_onnx_once(model_path='/home/zhengyuhang/multimodal-object-detection/RGBT-Detection/tools/rgbt_yolov5_op13_quantized.onnx', sample=[rgb_batch[0],t_batch[0]])
# onnx_result1 = infer_onnx_once(model_path='yolov5s_nc3_bs32_op13_onnxruntime_quantized.onnx', sample=benchmark_sample)

trt_result = trt_result[0].reshape(1,100800,6)
onnx_result0 = onnx_result0[0]
print(check(trt_result, onnx_result0, True), "max diff=%f"%(np.max(np.abs(trt_result - onnx_result0))) )
# onnx_result1 = onnx_result1[0]
# print(check(onnx_result0, onnx_result1, True), "max diff=%f"%(np.max(np.abs(onnx_result1 - onnx_result0))) )

# print(f'Time span (INT8 MODE): {tok - tick  : .4f} sec')
# print(f'Time span (INT8 MODE): {(tok - tick) / 512  : .4f} sec')