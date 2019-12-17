import numpy as np 
import time
import cv2
import pyrealsense2 as rs 
import random
import math

from matplotlib import pyplot as plt
from statistics import median

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

def find_mask_centre(mask, color_image):
    """
    Finding centre of mask and drawing a circle at the centre
    """
    moments = cv2.moments(np.float32(mask))

    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    return cX, cY

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

    return (masks, boxes, labels)
    

if __name__ == "__main__":

    cfg, predictor = create_predictor()

    # Configure video streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, RESOLUTION_X, RESOLUTION_Y, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, RESOLUTION_X, RESOLUTION_Y, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print("Depth Scale is: {:.4f}m".format(depth_scale))

    while True:
        
        time_start = time.time()
        
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        t1 = time.time()
        outputs = predictor(color_image)
        t2 = time.time()
        print("Model took {:.2f} time".format(t2 - t1))

        predictions = outputs['instances']

        if outputs['instances'].has('pred_masks'):
            num_masks = len(predictions.pred_masks)
        else:
            num_masks = 0
        
        v = Visualizer(color_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        
        masks, boxes, labels = format_results(predictions, v.metadata.get("thing_classes"))

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
            and depth range of (0 - MAX_RANGE millimeters). Iterate through each
            histogram bin until half of the mask area has been accounted for. 
            This is the median of the histogram and therefore the median depth.
            """
            mask_area = masks[i].area()
            num_median = math.floor(mask_area / 2)
            histg = cv2.calcHist([depth_image], [0], masks[i].mask, [NUM_BINS], [0, MAX_RANGE])
            
            counter = 0
            centre_depth = 0.0
            for x in range(len(histg)):
                counter += histg[x][0]
                if counter >= num_median:
                    # Half of histogram is iterated through,
                    # Therefore this bin contains the median
                    centre_depth = "{:.2f}m".format(x / 50)
                    break 
           
            #print("\nCOUNTER IS: {}".format(counter))
            #print("ACTUAL AREA: {}\n".format(mask_area))
        
            cX, cY = find_mask_centre(masks[i]._mask, v.output)
            v.draw_circle((cX, cY), (0, 0, 0))

            v.draw_text(centre_depth, (cX, cY + 20))
            
        
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('', v.output.get_image()[:, :, ::-1])
        #cv2.imshow('Depth', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time_end = time.time()
        total_time = time_end - time_start
        print("Time to process frame: {:.2f}".format(total_time))
        print("FPS: {:.2f}".format(1/total_time))
        
    pipeline.stop()
    cv2.destroyAllWindows()
