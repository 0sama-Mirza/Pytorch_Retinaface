from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import gc
import time

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

def detect_faces(
    trained_model_path,
    input_folder,
    output_folder,
    network='resnet50',
    confidence_threshold=0.02,
    top_k=5000,
    nms_threshold=0.4,
    keep_top_k=750,
    vis_thres=0.6,
    use_cpu=False,
    save_image=True
):
    def check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(state_dict, prefix):
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(model, pretrained_path, load_to_cpu):
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def resize_if_needed(img_raw, max_width=1920, max_height=1080):
        height, width = img_raw.shape[:2]
        if width <= max_width and height <= max_height:
            return img_raw
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv2.resize(img_raw, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img

    torch.set_grad_enabled(False)
    cfg = cfg_re50 if network == "resnet50" else cfg_mnet

    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, trained_model_path, use_cpu)
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cpu" if use_cpu else "cuda")
    net = net.to(device)
    resize = 1

    os.makedirs(output_folder, exist_ok=True)
    all_detections = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(input_folder, filename)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_raw is None:
            continue
        img_raw = resize_if_needed(img_raw)
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        with torch.no_grad():
            loc, conf, landms = net(img)
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([
                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                img.shape[3], img.shape[2]
            ])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]
            dets = dets[:keep_top_k, :]
            landms = landms[:keep_top_k, :]
            dets = np.concatenate((dets, landms), axis=1)
            # === Collect detection info ===
            image_result = {
                "Image": filename,
                "Number_of_faces_detected": len(dets),
                "Faces": []
            }
            for b in dets:
                if b[4] < vis_thres:
                    continue
                confidence = float(b[4])
                landmarks = {
                    "left_eye": (int(b[5]), int(b[6])),
                    "right_eye": (int(b[7]), int(b[8])),
                    "nose": (int(b[9]), int(b[10])),
                    "left_mouth": (int(b[11]), int(b[12])),
                    "right_mouth": (int(b[13]), int(b[14])),
                }
                image_result["Faces"].append({
                    "Confidence": confidence,
                    **landmarks
                })
                if save_image:
                    b_int = list(map(int, b))
                    cv2.rectangle(img_raw, (b_int[0], b_int[1]), (b_int[2], b_int[3]), (0, 0, 255), 2)
                    cx = b_int[0]
                    cy = b_int[1] + 12
                    text = "{:.4f}".format(b[4])
                    cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    cv2.circle(img_raw, (b_int[5], b_int[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b_int[7], b_int[8]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (b_int[9], b_int[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b_int[11], b_int[12]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b_int[13], b_int[14]), 1, (255, 0, 0), 4)
            all_detections.append(image_result)
            if save_image:
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img_raw)
            del img, loc, conf, landms, priors, boxes, scores, scale, scale1
            torch.cuda.empty_cache()
            gc.collect()
    return all_detections

def print_detection_summary(all_detections):
    """Prints the detection results in the specified format"""
    print("\n\n===== Detection Summary =====")
    for result in all_detections:
        print(f"\nImage: {result['Image']}")
        print(f"Number of faces detected: {result['Number_of_faces_detected']}")
        for i, face in enumerate(result["Faces"], start=1):
            print(f"Face {i}:")
            print(f"  Confidence: {face['Confidence']:.4f}")
            print(f"  left_eye: {face['left_eye']}")
            print(f"  right_eye: {face['right_eye']}")
            print(f"  nose: {face['nose']}")
            print(f"  left_mouth: {face['left_mouth']}")
            print(f"  right_mouth: {face['right_mouth']}")