"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed
import matplotlib.pyplot as plt  # Added import for plotting

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            boxes = pickle.load(f)
        return boxes

    with open(gt_path, 'r') as f:
        state = 0
        lines = f.readlines()
        lines = [line.rstrip('\r\n') for line in lines]
    boxes = {}
    print(f"Total lines in ground truth file: {len(lines)}")
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    with open(cache_file, 'wb') as f:
        pickle.dump(boxes, f)
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] == '':
            continue
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events, desc='Reading Predictions')

    for event in pbar:
        event_dir = os.path.join(pred_dir, event)
        if not os.path.isdir(event_dir):
            continue  # Skip if not a directory
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            if not imgtxt.endswith('.txt'):
                continue  # Skip non-txt files
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ Normalize scores to [0,1]
    pred {key: {img_name: [[x1,y1,x2,y2,s], ...]}}
    """

    max_score = 0
    min_score = 1

    for _, img_dict in pred.items():
        for _, boxes in img_dict.items():
            if len(boxes) == 0:
                continue
            _min = np.min(boxes[:, -1])
            _max = np.max(boxes[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score if max_score != min_score else 1  # Prevent division by zero
    for _, img_dict in pred.items():
        for _, boxes in img_dict.items():
            if len(boxes) == 0:
                continue
            boxes[:, -1] = (boxes[:, -1] - min_score) / diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ Single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap = gt_overlap.max()
        max_idx = gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        precision = pr_curve[i, 1] / pr_curve[i, 0] if pr_curve[i, 0] > 0 else 0
        recall = pr_curve[i, 1] / count_face if count_face > 0 else 0
        _pr_curve[i, 0] = precision
        _pr_curve[i, 1] = recall
    return _pr_curve


def voc_ap(rec, prec):

    # Correct AP calculation
    # First append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # To calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # And sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh=0.5):
    pred = get_preds(pred)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []

    # Dictionaries to store PR, F1, and Recall-Confidence curves for each setting
    pr_curves = {}
    f1_curves = {}
    recall_confidence_curves = {}

    # Generate the threshold array
    thresholds = 1 - (np.arange(thresh_num) + 1) / thresh_num  # Array of thresholds from ~0 to ~1

    for setting_id in range(3):
        # Different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [easy, medium, hard]
        pbar = tqdm.tqdm(range(event_num), desc=f'Processing {settings[setting_id]}')
        for i in pbar:
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred.get(event_name, {})
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                img_name = str(img_list[j][0][0])
                pred_info = pred_list.get(img_name, np.array([]))

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        precision = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, precision)
        aps.append(ap)

        # Compute F1 scores
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (precision * recall) / (precision + recall)
            f1 = np.nan_to_num(f1)  # Replace NaNs with zero

        # Store PR, F1, and Recall-Confidence curves for current setting
        pr_curves[settings[setting_id]] = (recall, precision)
        f1_curves[settings[setting_id]] = f1
        recall_confidence_curves[settings[setting_id]] = recall

    print("==================== Results ====================")
    for idx, setting in enumerate(settings):
        print(f"{setting.capitalize()} Val AP: {aps[idx]:.4f}")
    print("=================================================")

    # Plot Precision-Recall curves
    plt.figure(figsize=(12, 8))
    colors = {'easy': 'blue', 'medium': 'green', 'hard': 'red'}
    for setting in settings:
        recall, precision = pr_curves[setting]
        plt.plot(recall, precision, label=f"{setting.capitalize()} PR (AP: {aps[settings.index(setting)]:.4f})", color=colors[setting])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for WIDER FACE Evaluation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the PR plot as an image file
    pr_plot_path = 'precision_recall_curves.png'
    plt.savefig(pr_plot_path)
    print(f"Precision-Recall curves saved as '{pr_plot_path}'.")

    # Display the PR plot
    plt.show()

    # Plot F1 Confidence Curves
    plt.figure(figsize=(12, 8))
    for setting in settings:
        f1 = f1_curves[setting]
        plt.plot(thresh_num - np.arange(thresh_num), f1, label=f"{setting.capitalize()} F1", color=colors[setting])

    plt.xlabel('Threshold Index (Higher Index = Lower Confidence)')
    plt.ylabel('F1 Score')
    plt.title('F1 Confidence Curves for WIDER FACE Evaluation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the F1 plot as an image file
    f1_plot_path = 'f1_confidence_curves.png'
    plt.savefig(f1_plot_path)
    print(f"F1 Confidence curves saved as '{f1_plot_path}'.")

    # Display the F1 plot
    plt.show()

    # Plot Recall-Confidence Curves
    plt.figure(figsize=(12, 8))
    for setting in settings:
        recall = recall_confidence_curves[setting]
        plt.plot(thresholds, recall, label=f"{setting.capitalize()} Recall", color=colors[setting])

    plt.xlabel('Confidence Threshold')
    plt.ylabel('Recall')
    plt.title('Recall-Confidence Curves for WIDER FACE Evaluation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the Recall-Confidence plot as an image file
    recall_plot_path = 'recall_confidence_curves.png'
    plt.savefig(recall_plot_path)
    print(f"Recall-Confidence curves saved as '{recall_plot_path}'.")

    # Display the Recall-Confidence plot
    plt.show()

    # Optionally, plot all curves in a single figure using subplots
    # Uncomment the following block if you wish to have combined plots

    fig, axs = plt.subplots(3, 1, figsize=(12, 24))
    
    # Precision-Recall Curves
    for setting in settings:
        recall, precision = pr_curves[setting]
        axs[0].plot(recall, precision, label=f"{setting.capitalize()} PR (AP: {aps[settings.index(setting)]:.4f})", color=colors[setting])
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')
    axs[0].set_title('Precision-Recall Curves for WIDER FACE Evaluation')
    axs[0].legend()
    axs[0].grid(True)
    
    # F1 Confidence Curves
    for setting in settings:
        f1 = f1_curves[setting]
        axs[1].plot(thresh_num - np.arange(thresh_num), f1, label=f"{setting.capitalize()} F1", color=colors[setting])
    axs[1].set_xlabel('Threshold Index (Higher Index = Lower Confidence)')
    axs[1].set_ylabel('F1 Score')
    axs[1].set_title('F1 Confidence Curves for WIDER FACE Evaluation')
    axs[1].legend()
    axs[1].grid(True)
    
    # Recall-Confidence Curves
    for setting in settings:
        recall = recall_confidence_curves[setting]
        axs[2].plot(thresholds, recall, label=f"{setting.capitalize()} Recall", color=colors[setting])
    axs[2].set_xlabel('Confidence Threshold')
    axs[2].set_ylabel('Recall')
    axs[2].set_title('Recall-Confidence Curves for WIDER FACE Evaluation')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    combined_plot_path = 'combined_curves.png'
    plt.savefig(combined_plot_path)
    print(f"Combined curves saved as '{combined_plot_path}'.")
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="./widerface_txt/", help='Path to predictions directory')
    parser.add_argument('-g', '--gt', default='./ground_truth/', help='Path to ground truth directory')

    args = parser.parse_args()
    evaluation(args.pred, args.gt)
