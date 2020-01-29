import numpy as np 
import time
import cv2
import pyrealsense2 as rs 
import random
import math
import argparse

from matplotlib import pyplot as plt
from sort import *

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import ColorMode

from detectron2.data import MetadataCatalog

import torch, torchvision

# Resolution of camera streams
RESOLUTION_X = 1280
RESOLUTION_Y = 720

# Configuration for histogram for depth image
NUM_BINS = 500
MAX_RANGE = 10000


def create_predictor():
    """
    Setup config and return predictor. See config/defaults.py for more options
    """
    cfg = get_cfg()

    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # Mask R-CNN ResNet101 FPN weights
    cfg.MODEL.WEIGHTS = "model_final_a3ec72.pkl"
    # This determines the resizing of the image. At 0, resizing is disabled.
    cfg.INPUT.MIN_SIZE_TEST = 0

    return (cfg, DefaultPredictor(cfg))


def setup_image_config(video_file=None):
    """
    Setup config and video steams. If --file is specified as an argument, setup
    stream from file. The input of --file is a .bag file in the bag_files folder.
    .bag files can be created using d435_to_file in the tools folder.
    video_file is by default None, and thus will by default stream from the 
    device connected to the USB.
    """
    config = rs.config()

    if video_file is None:
        
        config.enable_stream(rs.stream.depth, RESOLUTION_X, RESOLUTION_Y, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, RESOLUTION_X, RESOLUTION_Y, rs.format.bgr8, 30)
        config.enable_record_to_file("output.bag")
    else:
        try:
            config.enable_device_from_file("bag_files/{}".format(video_file))
        except:
            print("Cannot enable device from: '{}'".format(video_file))

    return config


def format_results(predictions, class_names):
    """
    Format results so they can be used by overlay_instances function
    """
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None

    labels = None 
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

    masks = predictions.pred_masks.cpu().numpy()
    masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]

    boxes_list = boxes.tensor.tolist()
    scores_list = scores.tolist()

    for i in range(len(scores_list)):
        boxes_list[i].append(scores_list[i])
    
    boxes_list = np.asanyarray(boxes_list)
    #print(scores_list)
    #print(type(scores_list))
    print(boxes_list)
    #print(type(boxes_list))
    #print(scores)
    
    #hello = np.append(boxes_list, scores_list)
    #print(hello)
    return (masks, boxes, boxes_list, labels)
    

def find_mask_centre(mask, color_image):
    """
    Finding centre of mask and drawing a circle at the centre
    """
    moments = cv2.moments(np.float32(mask))

    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    return cX, cY


def find_median_depth(mask_area, num_median, histg):
    """
    Iterate through all histogram bins and stop at the median value. This is the
    median depth of the mask.
    """
    
    median_counter = 0
    centre_depth = "0.00"
    for x in range(0, len(histg)):
        median_counter += histg[x][0]
        if median_counter >= num_median:
            # Half of histogram is iterated through,
            # Therefore this bin contains the median
            centre_depth = "{:.2f}m".format(x / 50)
            break 

    return centre_depth

def debug_plots(color_image, depth_image, mask, histg, depth_colormap):
    """
    This function is used for debugging purposes. This plots the depth color-
    map, mask, mask and depth color-map bitwise_and, and histogram distrobutions
    of the full image and the masked image.
    """
    full_hist = cv2.calcHist([depth_image], [0], None, [NUM_BINS], [0, MAX_RANGE])
    masked_depth_image = cv2.bitwise_and(depth_colormap, depth_colormap, mask= mask)

    plt.figure()
            
    plt.subplot(2, 2, 1)
    plt.imshow(depth_colormap)

    plt.subplot(2, 2, 2)
    plt.imshow(masks[i].mask)

    plt.subplot(2, 2, 3).set_title(labels[i])
    plt.imshow(masked_depth_image)

    plt.subplot(2, 2, 4)
    plt.plot(full_hist)
    plt.plot(histg)
    plt.xlim([0, 600])
    plt.show()

if __name__ == "__main__":

    cfg, predictor = create_predictor()

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='type --file=file-name.bag to stream using file instead of webcam')
    args = parser.parse_args()
    
    config = setup_image_config(args.file)


    # Configure video streams
    pipeline = rs.pipeline()
    
    # Start streaming
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    mot_tracker = Sort()

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print("Depth Scale is: {:.4f}m".format(depth_scale))

    while True:
        
        time_start = time.time()
        
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        pad_x = max(color_image.shape[0] - color_image.shape[1], 0) * (RESOLUTION_Y / max(color_image.shape))
        pad_y = max(color_image.shape[1] - color_image.shape[0], 0) * (RESOLUTION_Y / max(color_image.shape))
        unpad_h = RESOLUTION_Y - pad_y
        unpad_w = RESOLUTION_Y - pad_x

        t1 = time.time()

        outputs = predictor(color_image)
        
        t2 = time.time()
        print("Model took {:.2f} time".format(t2 - t1))

        predictions = outputs['instances']

        if outputs['instances'].has('pred_masks'):
            num_masks = len(predictions.pred_masks)
        else:
            continue
        
        detectron_time = time.time()

        v = Visualizer(color_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        
        masks, boxes, boxes_list, labels = format_results(predictions, v.metadata.get("thing_classes"))
        """
        tracked_objects = mot_tracker.update(outputs['instances'])
        unique_labels = labels.unique()
        num_classes = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            cls = classes[int(cls_pred)]
            cv2.rectangle(color_image, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(color_image, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
            cv2.putText(color_image, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

        """
        #ret = np.asanyarray(outputs['instances'])
        #print(outputs['instances'])
        v.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=None,
            assigned_colors=None,
            alpha=0.3
        )
        
        
        for i in range(num_masks):
            """
            Converting depth image to a histogram with num bins of NUM_BINS 
            and depth range of (0 - MAX_RANGE millimeters)
            """
        
            mask_area = masks[i].area()
            num_median = math.floor(mask_area / 2)
            
            histg = cv2.calcHist([depth_image], [0], masks[i].mask, [NUM_BINS], [0, MAX_RANGE])
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Uncomment this to use the debugging function
            #debug_plots(color_image, depth_image, masks[i].mask, histg, depth_colormap)
            
            centre_depth = find_median_depth(mask_area, num_median, histg)
        
            cX, cY = find_mask_centre(masks[i]._mask, v.output)
            v.draw_circle((cX, cY), (0, 0, 0))
            v.draw_text(centre_depth, (cX, cY + 20))
            
        
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #cv2.imshow('Segmented Image', color_image)
        cv2.imshow('Segmented Image', v.output.get_image()[:, :, ::-1])
        #cv2.imshow('Depth', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time_end = time.time()
        total_time = time_end - time_start
        print("Time to process frame: {:.2f}".format(total_time))
        print("FPS: {:.2f}\n".format(1/total_time))
        
    pipeline.stop()
    cv2.destroyAllWindows()
    