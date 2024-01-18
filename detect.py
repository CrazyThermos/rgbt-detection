# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from model.frame import RGBTModel
from model.common import RGBTDetectMultiBackend
from dataset.rgbt_dataset import LoadRGBTImages
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, IMG_FORMATS, VID_FORMATS, check_file, check_img_size, colorstr, cv2, Annotator, save_one_box,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model

    device = select_device(device)
    backendmodel = RGBTModel(3, nc=1, gd=0.33,gw=0.5).to(device=device)
    backendmodel.names = ['person'] 
    model = RGBTDetectMultiBackend(backendmodel, weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # if webcam:
    #     view_img (warn=True)
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    #     bs = len(dataset)
    # elif screenshot:
    #     dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    # else:
    dataset = LoadRGBTImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    seen, windows = 0, [], 

    for rgb_path, t_path, im_rgb, im_t, im_rgb0s, im_t0s, vid_cap, s in dataset:
        # with dt[0]:
        im_rgb = torch.from_numpy(im_rgb).to(model.device)
        im_rgb = im_rgb.half() if model.fp16 else im_rgb.float()  # uint8 to fp16/32
        im_rgb /= 255  # 0 - 255 to 0.0 - 1.0
        im_t = torch.from_numpy(im_t).to(model.device)
        im_t = im_t.half() if model.fp16 else im_t.float()  # uint8 to fp16/32
        im_t /= 255  # 0 - 255 to 0.0 - 1.0
        
        if len(im_t.shape) == 3:
            im_rgb = im_rgb[None]  # expand for batch dim
            im_t = im_t[None]  # expand for batch dim

        # Inference
        # with dt[1]:
        visualize = increment_path(save_dir / Path(t_path).stem, mkdir=True) if visualize else False
        pred = model(im_rgb, im_t, augment=augment, visualize=visualize)

        # NMS
        # with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p_rgb, im_rgb0, frame = rgb_path[i], im_rgb0s[i].copy(), dataset.count
                p_t, im_t0, frame = t_path[i], im_t0s[i].copy(), dataset.count
                
                s += f'{i}: '
            else:
                p_rgb, im_rgb0, frame = rgb_path, im_rgb0s.copy(), getattr(dataset, 'frame', 0)
                p_t, im_t0, frame = t_path, im_t0s.copy(), getattr(dataset, 'frame', 0)

            p_rgb = Path(p_rgb)  # to Path
            p_t = Path(p_t)  # to Path

            rgb_save_path = str(save_dir / p_rgb.name)  # im.jpg
            # rgb_txt_path = str(save_dir / 'labels' / p_rgb.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            t_save_path = str(save_dir / p_t.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p_t.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            
            s += '%gx%g ' % im_t.shape[2:]  # print string
            # gn_rgb = torch.tensor(im_rgb0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc_rgb = im_rgb0.copy() if save_crop else im_rgb0  # for save_crop
            rgb_annotator = Annotator(im_rgb0, line_width=line_thickness, example=str(names))
            
            gn = torch.tensor(im_t0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc_t = im_t0.copy() if save_crop else im_t0  # for save_crop
            t_annotator = Annotator(im_t0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], im_t0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        rgb_annotator.box_label(xyxy, label)
                        t_annotator.box_label(xyxy, label)

                    if save_crop:
                        save_one_box(xyxy, imc_rgb, file=save_dir / 'crops' / names[c] / f'{p_rgb.stem}.jpg', BGR=True)
                        save_one_box(xyxy, imc_t, file=save_dir / 'crops' / names[c] / f'{p_t.stem}.jpg', BGR=True)


            # Stream results
            im_rgb0 = rgb_annotator.result()
            im_t0 = t_annotator.result()

            if view_img:
                if platform.system() == 'Linux' and p_rgb not in windows and p_t not in windows:
                    windows.append(p_rgb)
                    windows.append(p_t)
                    
                    cv2.namedWindow(str(p_rgb), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p_rgb), im_rgb0.shape[1], im_rgb0.shape[0])
                    cv2.namedWindow(str(p_t), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p_t), im_t0.shape[1], im_t0.shape[0])
                
                cv2.imshow(str(p_rgb), im_rgb0)
                cv2.imshow(str(p_t), im_t0)

                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(rgb_save_path, im_rgb0)
                    cv2.imwrite(t_save_path, im_t0)

                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im_t0.shape[1], im_t0.shape[0]
                            
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im_t0)

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
