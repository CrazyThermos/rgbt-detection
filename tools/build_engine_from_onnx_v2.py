# to be continue
import os
import json
import tensorrt as trt
import time
import numpy as np
import random
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Iterable, List, Union
from pathlib import Path
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt_version = [n for n in trt.__version__.split('.')]

class Config:
    def __init__(self, 
                 use_fp16 : bool = True, 
                 use_int8 : bool = True, 
                 use_strict : bool = False, 
                 use_qat : bool = False, 
                 use_sparsity : bool = False, 
                 timing_cache : str = 'timing.cache') -> None:
        self.use_fp16 = use_fp16
        self.use_int8 = use_int8
        # self.use_fc2_gemm = use_fc2_gemm
        self.use_strict = use_strict
        # self.is_calib_mode = False
        self.use_qat = use_qat
        self.use_sparsity = use_sparsity
        self.timing_cache = timing_cache
        

class RGBTCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, imgpath, img_format, calibration_count, calibrationCacheFile='rgbt.cache', shape=(2,3,640,640)):
        # trt.IInt8EntropyCalibrator2.__init__(self)
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = calibrationCacheFile
        self.batch_size = shape[0]
        self.Channel = shape[1]
        self.height = shape[3]
        self.width = shape[2]
        self.input_shape = shape
        self.calibration_count = calibration_count
        self.imgs = self.load_rgbt_calibration_dataset(directory = imgpath, img_format = img_format)

        self.batch_idx = 0
        self.max_batch_idx = calibration_count // self.batch_size
        self.calibration_data = np.zeros((self.batch_size, 3, self.height, self.width), dtype=np.float32)
        self.data_size = self.calibration_data.nbytes
        self.device_input = cuda.mem_alloc(self.data_size)
        self.device_input_rgb = cuda.mem_alloc(self.data_size)
        self.device_input_t = cuda.mem_alloc(self.data_size)

    def load_rgbt_calibration_dataset(self, directory: str, img_format: str = 'jpg'):
        path = Path(directory)
        rgb_images = sorted(list(path.glob('*_rgb.'+img_format)))
        t_images = sorted(list(path.glob('*_t.'+img_format)))
        assert len(rgb_images) == len(t_images), f'The number of RGB images is not equal to the number of infrared images, {len(rgb_images)}!={len(t_images)}'
        pair_list = list(zip(rgb_images, t_images))
        random_list = random.sample(pair_list, self.calibration_count)
        return random_list

    def load_image(self, image_path:str):
        h, w= self.input_shape[2], self.input_shape[3]
        im = cv2.imread(str(image_path))
        im = cv2.resize(im, (h, w), interpolation=cv2.INTER_AREA)
        return im

    def free(self):
        self.device_input.free()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            if self.input_shape[0] == 2:
                batch_imgs = self.next_batch()
                if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.Channel * self.height * self.width:
                    return None
                cuda.memcpy_htod(self.device_input, batch_imgs)
                return [self.device_input]
            else :
                rgb_batch_imgs, t_batch_imgs = self.next_batch()
                cuda.memcpy_htod(self.device_input_rgb, rgb_batch_imgs)
                cuda.memcpy_htod(self.device_input_t, t_batch_imgs)
                return [self.device_input_rgb, self.device_input_t]
        except:
            print('wrong')
            return None
        
    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size: \
                                    (self.batch_idx + 1) * self.batch_size]

            rgb_batch_imgs = np.zeros((self.batch_size, self.Channel, self.height, self.width),
                        dtype=np.float32)
            t_batch_imgs = np.zeros((self.batch_size, self.Channel, self.height, self.width),
                        dtype=np.float32)
            i = 0
            for rgb_image, t_image in batch_files:
                if i < self.batch_size:
                    rgb_tensor = self.load_image(str(rgb_image)).transpose((2, 0, 1))[::-1, :, :]
                    t_tensor = self.load_image(str(t_image)).transpose((2, 0, 1))[::-1, :, :]
                    rgb_tensor = np.ascontiguousarray(rgb_tensor).astype(np.float32) / 255.0
                    t_tensor = np.ascontiguousarray(t_tensor).astype(np.float32) / 255.0

                    rgb_batch_imgs[i] = rgb_tensor
                    t_batch_imgs[i] = t_tensor
                    i += 1
            self.batch_idx += 1
            if self.input_shape[0] == 2:
                batch_imgs = np.zeros((self.batch_size, self.Channel, self.height, self.width),
                            dtype=np.float32)
                batch_imgs[0] = rgb_tensor
                batch_imgs[1] = t_tensor
                return np.ascontiguousarray(batch_imgs)
            return (np.ascontiguousarray(rgb_batch_imgs), np.ascontiguousarray(t_batch_imgs))
        else:
            return np.array([])
        
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            # os.fsync(f)

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

def build_engine(onnx_file, 
                 engine_file,
                 workspace_size, 
                 config,  
                 imgpath,
                 img_shape,
                 calibrationCacheFile, 
                 calib_num, 
                 verbose,
                 int8_scale_file):

    network_creation_flag = 0
    if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys():
        network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_creation_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        builder_config.avg_timing_iterations = 8
        # Cublas tactics can be unset once the qkv plugin does not use it anymore.
        builder_config.set_tactic_sources(builder_config.get_tactic_sources() | 1 << int(trt.TacticSource.CUBLAS))
        if config.use_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if config.use_int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
            if not config.use_qat:
                calibrator = RGBTCalibrator(imgpath=imgpath, img_format = 'png', calibration_count=calib_num, calibrationCacheFile=calibrationCacheFile, shape=img_shape)
                # calibrator.get_batch(None,None)
                builder_config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
                builder_config.int8_calibrator = calibrator
        if config.use_strict:
            builder_config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            builder_config.set_flag(trt.BuilderFlag.DIRECT_IO)
            builder_config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

        if verbose:
            builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        if config.use_sparsity:
            TRT_LOGGER.log(TRT_LOGGER.INFO, "Setting sparsity flag on builder_config.")
            builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # speed up the engine build for trt major version >= 8
        # 1. disable cudnn tactic
        # 2. load global timing cache
        if int(trt_version[0]) >= 8:
            tactic_source = builder_config.get_tactic_sources() & ~(1 << int(trt.TacticSource.CUDNN))
            builder_config.set_tactic_sources(tactic_source)
            if config.timing_cache != None:
                if os.path.exists(config.timing_cache):
                    with open(config.timing_cache, "rb") as f:
                        cache = builder_config.create_timing_cache(f.read())
                        builder_config.set_timing_cache(cache, ignore_mismatch = False)
                else:
                    cache = builder_config.create_timing_cache(b"")
                    builder_config.set_timing_cache(cache, ignore_mismatch = False)

        # only use the largest sequence when in calibration mode

        # Create the network
        parser = trt.OnnxParser(network, TRT_LOGGER)
        if not os.path.exists(onnx_file):
            raise FileNotFoundError(f'ONNX file {onnx_file} not found')

        with open(onnx_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        if int8_scale_file is not None and config.use_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            setDynamicRange(network, int8_scale_file)

        build_start_time = time.time()
        serialized_engine = builder.build_serialized_network(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))

        # save global timing cache
        if int(trt_version[0]) >= 8 and config.timing_cache != None:
            cache = builder_config.get_timing_cache()
            with cache.serialize() as buffer:
                with open(config.timing_cache, "wb") as f:
                    f.write(buffer)
                    f.flush()
                    os.fsync(f)

        if config.use_int8 and not config.use_qat:
            calibrator.free()
        # return serialized_engine
        with open(engine_file, "wb") as f:
            f.write(serialized_engine)

if __name__ == "__main__":
        config = Config()
        build_engine(
                 onnx_file = 'rgbt_ca_rtdetrv2_589_m3fd_op18.onnx', 
                 engine_file = 'rgbt_ca_rtdetrv2_589_m3fd_int8_SYMM_LINEAR_PERCHANNEL.engine', 
                 workspace_size = 4294967296<<4, 
                 config = config,  
                 imgpath = '/home/zhengyuhang/datasets/M3FD_yolo/images/val',
                 img_shape = (1, 3, 640, 640),
                 calibrationCacheFile = 'rgbt.cache', 
                 calib_num=512, 
                 verbose = True,
                 int8_scale_file=None )