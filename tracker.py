import sys
from pathlib import Path
import numpy as np
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import (non_max_suppression, scale_boxes, check_img_size)
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from trackers.multi_tracker_zoo import create_tracker


def load_detector(weights=ROOT/'weights'/'yolov5s.pt', device='', imgsz=640, dnn=False, half=False):
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = [imgsz, imgsz] if type(imgsz) == int else imgsz

    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    if half:
        model.half()

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    return model, stride, names, pt, device


def load_tracker(nr_sources=1,
                 tracking_method='strongsort',
                 reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                 half=False,
                 device=''
                 ):
    if device == '':
        device = select_device(device)
    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources
    return tracker_list, outputs


@torch.no_grad()
def detect(model,
        names,
        image,
        tracker_list,
        outputs,
        device='',
        line_thickness=2,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic_nms=False,
        max_det=1000,
        draw_detections=False,
        imgsz=640,
        ):

    imgsz = [imgsz, imgsz] if type(imgsz) == int else imgsz
    # device = select_device(device)
    im = letterbox(image, imgsz, stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    if draw_detections:
        annotator = Annotator(image, line_width=line_thickness, example=str(names))
    s = ""
    for i, det in enumerate(pred):
        s += '%gx%g ' % im.shape[2:]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()  # xyxy

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                outputs[i] = tracker_list[i].update(det.cpu(), image)

                # draw boxes for visualization
                if draw_detections and len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                        bboxes = output[0:4]
                        id = int(output[4])
                        cls = int(output[5])
                        label = f'{id} {names[cls]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(cls, True))

    print(s)
    return outputs
