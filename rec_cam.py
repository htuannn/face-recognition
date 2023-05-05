import argparse
import time
from pathlib import Path
import sys
import os
import json
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from numpy import random
import copy

from collections import Counter

from face_embedding.models.model import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5n-face.pt', help='model.pt path(s)')
parser.add_argument('--backbone', default='iresnet18', type=str,
                    help='backbone architechture')
parser.add_argument('--pretrained_backbone', default=False, type=str,
                    help='Use pretrain backbone')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--feat_list', default='data_recognition/preprocessed/face_embdding.txt', type=str,
                    help='Path for saving features file')
parser.add_argument('--label_map', default='data_recognition/preprocessed/label_map.json', type=str,
                    help='Path for saving label dictionary file')
parser.add_argument('--embedding-size', default=512, type=int,
                    help='The embedding feature size')

parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--save-img', action='store_true', help='save results')
parser.add_argument('--view-img', action='store_true', help='show results')
parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')    
args = parser.parse_args()


from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


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

def show_results(img, xyxy, conf, identify):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]


    tf = max(tl - 1, 1)  # font thickness
    conf = str(conf)[:5]
    cv2.putText(img, str(identify), (x1 + 30, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(img, conf, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def process(
    detect_model,
    embd_model,
    source,
    device,
    project,
    name,
    feat_list,
    label_map,
    exist_ok,
    save_img,
    view_img
):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz=(640, 640)
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    
    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    for path, im, im0s, vid_cap in dataset:
        
        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
        else:
            orgimg = im.transpose(1, 2, 0)
        
        orgimg = cv2.cvtColor(cv2.flip(orgimg,1), cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=detect_model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = detect_model(img)[0]
        # detections = (pred)
        # print('previous',pred[:,:4])
        # break
        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    
        # print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            im0=cv2.flip(im0,1)
            p = Path(p)  # to Path
            save_path = str(Path(save_dir) / p.name)  # im.jpg
            # print(det)
            if len(det):
                # Rescale boxes from img_size to im0 size
                #print('previos', det[:, :4])
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print(det[:, :4])
                # Print results
               
                #print('after', det[:, : 4])
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                # print(det[:, :4])
                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                transform = transforms.Compose([
			        transforms.ToPILImage(),  # Chuyển đổi ảnh thành đối tượng PIL Image
			        transforms.Resize((112, 112)), 
			        transforms.ToTensor(),  # Chuẩn hóa giá trị pixel
			    ])
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    x1,y1,x2,y2= xyxy
                    #crop face 
                    #face = cv2.cvtColor(im0[int(y1):int(y2),int(x1):int(x2)], cv2.COLOR_BGR2RGB)
                    face = im0[int(y1):int(y2),int(x1):int(x2)]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = transform(face).unsqueeze(0).to(device)
                    _emb_feat = embd_model(face)
                    _emb_feat = _emb_feat.squeeze().data.cpu().numpy()
                    _label, _dict=majority_vote_simmilar(args, _emb_feat, feat_list, label, 1)

                    if label_map is None:
                        _iden= _label
                    else:
                        _iden =label_map[str(_label)]
                        
                    im0 = show_results(im0, xyxy, _dict , _iden)
            
            if view_img:
                cv2.imshow('result', im0)
                k = cv2.waitKey(1)
                    
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
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
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0)
                    except Exception as e:
                        print(e)
def get_embdding_feature(args):
    label= []
    feat= []
    with open(args.feat_list, 'r') as f:
        file= f.read()
        for line in file.split('\n')[:-1]:
            arr=line.split()
            label.append(arr[0])
            feat.append(np.array(arr[1:]).astype('float64'))
            
    return np.array(feat), np.array(label)

def cosine_similarity(args, face1_feat, face2_feat):
    # Compute the cosine similarity
    face1_feat= torch.tensor(face1_feat).float().to(args.device)
    face2_feat= torch.tensor(face2_feat).float().to(args.device)
    dot = torch.dot(face1_feat, face2_feat)
    norma = torch.linalg.norm(face1_feat)
    normb = torch.linalg.norm(face2_feat)
    cos = dot / (norma * normb)
    return cos.cpu().numpy()

def majority_vote_simmilar(args, _feat, feat_list, labels, k):
    # Compute cosine similarity between each training and test data points
    similarities = np.apply_along_axis(lambda x: cosine_similarity(args, _feat, x), 1, feat_list)
    #print(similarities)
    # Get indices of k nearest neighbors for each test data point
    k_indices = np.apply_along_axis(lambda x: np.argsort(x)[-k:], 0, similarities)
    
    # Get labels of k nearest neighbors for each test data point
    k_labels = np.apply_along_axis(lambda x: labels[x], 0, k_indices)

    _pred = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 0, k_labels)
    _distance= np.apply_along_axis(lambda x: np.sort(x)[-k:], 0, similarities)[k_labels == _pred]

    return _pred, sum(_distance)/len(_distance)

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        with open(args.label_map, 'r') as f:
            label_map= json.load(f)
    except:
        label_map=None


    feat, label= get_embdding_feature(args)
    #clf= SVC(kernel='linear', probability=True, C=1.)

    args.device=torch.device("cuda" if (not args.cpu_mode) &(torch.cuda.is_available()) else "cpu") 

    detect_model = load_model(args.weights, args.device)


    embd_model= load_dict_inf(args, build_backbone(args).to(args.device))
    embd_model.eval()
    process(detect_model, embd_model, args.source, args.device, args.project, args.name, feat, label_map, args.exist_ok, args.save_img, args.view_img)
