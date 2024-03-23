"""
This is a highly flexible PPQ quantization entry script, 
    and you will witness the power of PPQ as an offline neural network quantization tool.

PPQ abstracts neural network quantization into several parts, such as 
    Quantizer,                  (ppq.quantization.quantizers) 
    Optimization Pass,          (ppq.quantization.optim)
    Optimization Pipeline,      (ppq.quantization.optim)
    Exporter,                   (ppq.parser)
    Tensor Quantization Config, (ppq.core)
    etc.

In this example, we will create them one by one to customize the entire quantization logic. 
    We will customize the quantization rules and use a custom optimization process to fine-tune the quantization rules in detail. 
    We will create a custom exporter to print all quantization information to the screen.

"""

import os
import random
import numpy as np
import cv2 as cv

import torch
import torchvision

import ppq.lib as PFL
from ppq import (BaseGraph, BaseQuantizer, GraphExporter, Operation,
                 OperationQuantizationConfig, QuantableOperation,
                 QuantableVariable, QuantizationOptimizationPass, SearchableGraph,
                 QuantizationPolicy, QuantizationProperty, QuantizationStates, RoundingPolicy,
                 TargetPlatform, TorchExecutor, graphwise_error_analyse, layerwise_error_analyse)
from ppq.api import ENABLE_CUDA_KERNEL, load_torch_model, load_onnx_graph, export_ppq_graph
from ppq.IR import GraphCommandProcessor, SearchableGraph

from ppq.quantization.optim import *
from typing import Iterable, List, Union
from pathlib import Path

class CustomizedExporter(GraphExporter):
    def export(self, file_path: str, save_name: str, platform: int, graph: BaseGraph, config_path: str = None, show_config: bool = False, **kwargs):
        print('This exporter does not export quantitative information to file, '
              'it prints quantitative information to the console instead.')
        for opname, op in graph.operations.items():
            # Skip those operators that are not involved in quantization.
            # They do not have a quantization configuration.
            if not isinstance(op, QuantableOperation): continue

            if show_config:
                print(f'### Quantization Configuration of {opname}: ')
                for idx, config in enumerate(op.config.input_quantization_config):
                    print(f'\t #### Input {idx}: ')
                    print(f'\t Scale: {config.scale.tolist()}')
                    print(f'\t Offset: {config.offset.tolist()}')
                    print(f'\t State: {config.state}')
                    print(f'\t Bitwidth: {config.num_of_bits}')
                    print(f'\t Quant_min: {config.quant_min}')
                    print(f'\t Quant_max: {config.quant_max}')
                
                for idx, config in enumerate(op.config.output_quantization_config):
                    print(f'\t #### Output {idx}: ')
                    print(f'\t Scale: {config.scale.tolist()}')
                    print(f'\t Offset: {config.offset.tolist()}')
                    print(f'\t State: {config.state}')
                    print(f'\t Bitwidth: {config.num_of_bits}')
                    print(f'\t Quant_min: {config.quant_min}')
                    print(f'\t Quant_max: {config.quant_max}')

        export_ppq_graph(
            graph=graph, platform=platform,
            graph_save_to = os.path.join(file_path, save_name+'_quantized.onnx'),
            config_save_to = os.path.join(file_path, save_name+'_quantized.json'))

class CustomizedInt8Quantizer(BaseQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, GraphCommandProcessor]
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = - 128
        self._quant_max = + 127

    def init_quantize_config(
        self, operation: Operation) -> OperationQuantizationConfig:
        OQC = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul', 'PPQBiasFusedMatMul'}:
            # base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv', 'ConvTranspose'}:
                if operation.inputs[1].is_parameter:
                    conv_weight_config = OQC.input_quantization_config[1]
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    conv_weight_config.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)
                    conv_weight_config.observer_algorithm = 'minmax'

            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm', 'MatMul', 'PPQBiasFusedMatMul'}:
                if operation.inputs[1].is_parameter:
                    gemm_weight_config = OQC.input_quantization_config[1]
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    gemm_weight_config.channel_axis = 0
                    gemm_weight_config.observer_algorithm = 'minmax'

            if operation.num_of_input > 2:
                bias_config = OQC.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        if operation.type == 'LayerNormalization':
            # Layernorm - gamma and beta need to be FP32
            for TQC in OQC.input_quantization_config[1: ]:
                TQC.state = QuantizationStates.FP32

        if operation.type == 'Attention':
            # Attention - Only input and weight need to be quantized.
            for TQC in OQC.input_quantization_config[2: ]:
                TQC.state = QuantizationStates.FP32

        return OQC

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.TRT_INT8

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool', 'Softmax',
            'Mul', 'Add', 'Max', 'Sub', 'Div', 'Reshape',
            'LeakyRelu', 'Concat', 'Sigmoid', 'Interp',
            'ReduceMean', 'Transpose', 'Slice', 'Flatten',
            'HardSwish', 'HardSigmoid', 'MatMul',
            'Attention', 'LayerNormalization', 'Gelu',
            'PPQBiasFusedMatMul'
        }

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip', 'Swish', 'Clip', 'SoftPlus', 'Sigmoid', 'Gelu'}

class CustomizedOptimPass(QuantizationOptimizationPass):
    """
    This custom Optimization Pass will perform a series of customized quantization.
    This is an example code, and you need to carefully read the code definition of the 
        Optimization Pass and understand how to control the quantization logic through the code.

    This Optimization Pass will:
        1. fuse relu - clip structure.
        2. set clip output scale in the network to 1/127.
        3. set the input and output quantization information of the abs operators to be the same.
        4. modify calibration method for some operators.
    """
    def __init__(self, name: str = 'My Optim Pass') -> None:
        super().__init__(name)

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        # fuse relu - clip, set output scale of relu to be 1/127
        processor = SearchableGraph(graph)
        patterns = processor.pattern_matching(
            patterns=['Relu', 'Clip'],
            edges=[[0, 1]], exclusive=True)

        for relu, clip in patterns:
            print(f'Fuse {relu.name} and {clip.name}')
            if not isinstance(clip, QuantableOperation): continue
            if not isinstance(relu, QuantableOperation): continue
            relu.config.output_quantization_config[0].dominated_by = (
                clip.config.output_quantization_config[0])
            clip.config.input_quantization_config[0].dominated_by = (
                clip.config.output_quantization_config[0])
            clip.config.output_quantization_config[0].scale = torch.tensor(1 / 127).cuda()
            clip.config.output_quantization_config[0].offset = torch.tensor(0.0).cuda()
            clip.config.output_quantization_config[0].state = QuantizationStates.ACTIVATED

        # keep input and output scale of abs as the same.
        for op in graph.operations.values():
            print(f'Op {op.name} has processed.')
            if op.type != 'Abs': continue
            if (isinstance(op, QuantableOperation)):
                ITQC = op.config.input_quantization_config[0]
                OTQC = op.config.output_quantization_config[0]
                ITQC.dominated_by = OTQC

        # modify calibration methods.
        for op in graph.operations.values():
            if op.name != 'My Op': continue
            if (isinstance(op, QuantableOperation)):
                ITQC = op.config.input_quantization_config[0]
                OTQC = op.config.output_quantization_config[0]
                ITQC.observer_algorithm = 'kl'
                OTQC.observer_algorithm = 'mse'

def load_image(image_path:str, input_shape:List[int]):
    h, w= input_shape[2],input_shape[3]
    im = cv.imread(str(image_path))
    im = cv.resize(im, (h, w), interpolation=cv.INTER_AREA)
    return im

def load_calibration_dataset( directory: str, input_shape: List[int], batch_size: int, calibration_count: int) -> Iterable:
    # ------------------------------------------------------------
    # 让我们从创建 calibration 数据开始做起， PPQ 需要你送入 32 ~ 1024 个样本数据作为校准数据集
    # 它们应该尽可能服从真实样本的分布，量化过程如同训练过程一样存在可能的过拟合问题
    # 你应当保证校准数据是经过正确预处理的、有代表性的数据，否则量化将会失败；校准数据不需要标签；数据集不能乱序
    # ------------------------------------------------------------
    random_list=[]
    images_list=[]
    dir_paths = Path(directory)
    images = sorted(list(dir_paths.glob('*.jpg')))
    images_list += list(images)
    random_list = random.sample(images_list, calibration_count)
    batches, batch = [], []
    

    for image in random_list:
        if len(batch) < batch_size:
            image_tensor = torch.from_numpy(load_image(str(image), input_shape=input_shape)).unsqueeze(0).permute(0,3,1,2)
            batch.append(image_tensor)
        else:
            batches.append(torch.cat(batch, dim=0))
            image_tensor = torch.from_numpy(load_image(str(image), input_shape=input_shape)).unsqueeze(0).permute(0,3,1,2)
            batch = [image_tensor]
    if len(batch) != 0:
        batches.append(torch.cat(batch, dim=0))

    return list(batches)

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE).float() / 255

def load_calibration_rgbt_dataset( directory: str, input_shape: List[int], batch_size: int, calibration_count: int) -> Iterable:
    # ------------------------------------------------------------
    # 让我们从创建 calibration 数据开始做起， PPQ 需要你送入 32 ~ 1024 个样本数据作为校准数据集
    # 它们应该尽可能服从真实样本的分布，量化过程如同训练过程一样存在可能的过拟合问题
    # 你应当保证校准数据是经过正确预处理的、有代表性的数据，否则量化将会失败；校准数据不需要标签；数据集不能乱序
    # ------------------------------------------------------------
    path = Path(directory)
    rgb_images = sorted(list(path.glob('*_rgb.jpg')))
    t_images = sorted(list(path.glob('*_t.jpg')))
    assert len(rgb_images) == len(t_images), f'The number of RGB images is not equal to the number of infrared images, {len(rgb_images)}!={len(t_images)}'
    pair_list = list(zip(rgb_images, t_images))
    random_list = random.sample(pair_list, calibration_count)
    # rgb_sample_list, t_sample_list = zip(*random_list)
    rgb_batches, rgb_batch, t_batches, t_batch = [], [], [], []
    

    for rgb_image, t_image in random_list:
        if len(t_batch) < batch_size:
            rgb_tensor = torch.from_numpy(load_image(str(rgb_image), input_shape=input_shape)).unsqueeze(0).permute(0,3,1,2)
            t_tensor = torch.from_numpy(load_image(str(t_image), input_shape=input_shape)).unsqueeze(0).permute(0,3,1,2)
            rgb_batch.append(rgb_tensor)
            t_batch.append(t_tensor)
        else:
            rgb_batches.append(torch.cat(rgb_batch, dim=0))
            t_batches.append(torch.cat(t_batch, dim=0))
            rgb_tensor = torch.from_numpy(load_image(str(rgb_image), input_shape=input_shape)).unsqueeze(0).permute(0,3,1,2)
            t_tensor = torch.from_numpy(load_image(str(t_image), input_shape=input_shape)).unsqueeze(0).permute(0,3,1,2)
            rgb_batch = [rgb_tensor]
            t_batch = [t_tensor]
    if len(t_batch) != 0:
        rgb_batches.append(torch.cat(rgb_batch, dim=0))
        t_batches.append(torch.cat(t_batch, dim=0))

    return list(zip(rgb_batches,t_batches))

def rgbt_collate_fn(batch: torch.Tensor) -> torch.Tensor:
    rgb_batches, t_batches = batch[0], batch[1]
    return [rgb_batches.to(DEVICE).float() / 255, t_batches.to(DEVICE).float() / 255]


if __name__ == "__main__":
    BATCHSIZE   = 1
    INPUT_SHAPE = [BATCHSIZE, 3, 1280, 1280]
    DEVICE      = 'cuda'
    QUANT_PLATFORM    = TargetPlatform.TRT_INT8
    DEPLOY_PLATFORM    = TargetPlatform.TRT_INT8

    EXPORT_DIRECTORY = './'
    DATASET_DIRECTORY = '/home/zhengyuhang/datasets/LLVIP_yolo/images/val'
    ONNX_PATH = './rgbt_yolov5_op13.onnx'
    CALIBRATION_COUNT = 128
    REQUIRE_ANALYSE = True
    RGBT_MODEL = True

    with ENABLE_CUDA_KERNEL():
        # torch model
        # model = torchvision.models.resnet18(pretrained=True).cuda()
        # graph = load_torch_model(model=model, sample=torch.zeros(size=[1, 3, 224, 224]).cuda())

        graph = load_onnx_graph(ONNX_PATH)
        quantizer   = CustomizedInt8Quantizer(graph=graph)

        dispatching = PFL.Dispatcher(graph=graph).dispatch(
            quant_types=quantizer.quant_operation_types)

        search_engine = SearchableGraph(graph)
        
        # operators with low quantization accuracy are not quantized
        for op in search_engine.opset_matching( 
            sp_expr=lambda x: x.name in {'/model.24/m.0/Conv','/model.24/m.1/Conv','/model.24/m.2/Conv'},
            rp_expr=lambda x, y: True,
            ep_expr=None, direction='down'
        ):
            dispatching[op.name] = TargetPlatform.FP32

        # initialize quantization information
        for op in graph.operations.values():
            quantizer.quantize_operation(
                op_name = op.name, platform = dispatching[op.name])

        executor = TorchExecutor(graph=graph, device='cuda')
        executor.tracing_operation_meta(inputs=[torch.zeros(size=INPUT_SHAPE).cuda(),torch.zeros(size=INPUT_SHAPE).cuda()])
        executor.load_graph(graph=graph)

        # Manually create a quantization optimization pipeline.
        pipeline = PFL.Pipeline([
            QuantizeSimplifyPass(),
            QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
            ParameterQuantizePass(),
            # CustomizedOptimPass(),                                         # <----- Insert Our Optimization Pass 
            RuntimeCalibrationPass(),
            PassiveParameterQuantizePass(),
            QuantAlignmentPass(force_overlap=True),
            # 微调你的网路
            LearnedStepSizePass(steps=1500),
            # 如果需要训练微调网络，训练过程必须发生在 ParameterBakingPass 之前
            ParameterBakingPass()
        ])

        if RGBT_MODEL:
            calibration_dataloader = load_calibration_rgbt_dataset(directory=DATASET_DIRECTORY, input_shape=INPUT_SHAPE, batch_size=BATCHSIZE, calibration_count=CALIBRATION_COUNT)
            pipeline.optimize(
                graph=graph, dataloader=calibration_dataloader, verbose=True, 
                calib_steps=CALIBRATION_COUNT, collate_fn=rgbt_collate_fn, executor=executor)
            
            graphwise_error_analyse(
                graph=graph, running_device='cuda', dataloader=calibration_dataloader, 
                collate_fn=rgbt_collate_fn)
            
            if REQUIRE_ANALYSE:
                print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
                layerwise_error_analyse(graph=graph, running_device=DEVICE,
                                        interested_outputs=None,
                                        dataloader=calibration_dataloader, collate_fn=rgbt_collate_fn)
            CustomizedExporter().export(file_path=EXPORT_DIRECTORY, save_name='rgbt_yolov5_op13', 
                                platform=DEPLOY_PLATFORM, graph=graph, config_path=None)
        else:
            # Calling quantization optimization pipeline.
            calibration_dataloader = load_calibration_dataset(directory=DATASET_DIRECTORY, input_shape=INPUT_SHAPE, batch_size=BATCHSIZE, calibration_count=CALIBRATION_COUNT)
            pipeline.optimize(
                graph=graph, dataloader=calibration_dataloader, verbose=True, 
                calib_steps=CALIBRATION_COUNT, collate_fn=collate_fn, executor=executor)
            
            graphwise_error_analyse(
                graph=graph, running_device='cuda', dataloader=calibration_dataloader, 
                collate_fn=collate_fn)
            
            if REQUIRE_ANALYSE:
                print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
                layerwise_error_analyse(graph=graph, running_device=DEVICE,
                                        interested_outputs=None,
                                        dataloader=calibration_dataloader, collate_fn=collate_fn)
            CustomizedExporter().export(file_path=EXPORT_DIRECTORY, save_name='rgbt_yolov5_op13', 
                                platform=DEPLOY_PLATFORM, graph=graph, config_path=None)
