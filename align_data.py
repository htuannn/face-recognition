import argparse
import pprint
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from face_embedding.dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5n-face.pt', help='model.pt path(s)')
parser.add_argument('--folderdataset_dir', default='data_recognition/raw', type=str,
                    help='Path to Face Image Folder Dataset')
parser.add_argument('--save_path', default='data_recognition/preprocessed', type=str,
                    help='Path for saving folder')
parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
parser.add_argument('--conf_thres', default=0.6, type=float,
                    help='')
parser.add_argument('--iou_thres', default=0.5, type=float,
                    help='')

args=parser.parse_args()

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def main(args):
    device=torch.device("cuda" if (not args.cpu_mode) &(torch.cuda.is_available()) else "cpu") 

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((640, 640)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                         std=[1., 1., 1.]),
    ])    
    data_inf = FaceImageFolderDataset(root= args.folderdataset_dir, transform = transform)

    model = load_model(args.weights, device)

    for idx, (img, _) in enumerate(data_inf):
        img=img.to(device)
        pred = model(img.unsqueeze(0))[0]
        pred = non_max_suppression_face(pred, args.conf_thres, args.iou_thres)[0]
        if len(pred) !=1:
            logging.warning(f"Image{data_inf.img_paths[idx]} error (no face or more than 1 face) !")
            continue
        #rescale bb to orginial size
        im0= cv2.imread(data_inf.img_paths[idx])
        pred[:, :4] = scale_coords(img.shape[1:], pred[:, :4], im0.shape).round()
        
        xyxy = pred[0, :4].view(-1).tolist()
        x1,y1,x2,y2 = xyxy

        cropped_face= im0[int(y1):int(y2),int(x1):int(x2)]
        save_path=os.path.join(args.save_path, data_inf.img_ids[idx])
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(os.path.join(save_path, data_inf.img_paths[idx].split('\\')[-1]), cropped_face)
        


if __name__ == '__main__':
    pprint.pprint(vars(args))
    main(args)
