import numpy as np 
import time
import cv2
import pyrealsense2 as rs 
import random
import math
import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
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

img_size = 416

"""
class AssociatedDetection:

    def __init__(self):
"""

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

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
    class_list = classes.tolist()

    for i in range(len(scores_list)):
        boxes_list[i].append(scores_list[i])
        boxes_list[i].append(class_list[i])
    
    #print(boxes_list)
    boxes_list = np.array(boxes_list)
    #print(scores_list)
    #print(type(scores_list))
    #print(boxes_list)
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
            centre_depth = x / 50
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

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # Initialise Kalman filter tracker from Sort
    mot_tracker = Sort()

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print("Depth Scale is: {:.4f}m".format(depth_scale))

    speed_time_start = time.time()

    while True:
        
        time_start = time.time()
        
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        print(color_image.shape)

        pad_x = max(color_image.shape[0] - color_image.shape[1], 0) * (RESOLUTION_X / max(color_image.shape))
        pad_y = max(color_image.shape[1] - color_image.shape[0], 0) * (360 / max(color_image.shape))
        unpad_h = RESOLUTION_Y - pad_y
        unpad_w = RESOLUTION_X - pad_x

        t1 = time.time()

        outputs = predictor(color_image)
        
        t2 = time.time()
        print("Model took {:.2f} time".format(t2 - t1))

        predictions = outputs['instances']

        if outputs['instances'].has('pred_masks'):
            num_masks = len(predictions.pred_masks)
        else:
            tracked_objects = mot_tracker.update(boxes_list)
            continue
        
        detectron_time = time.time()

        v = Visualizer(color_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        
        masks, boxes, boxes_list, labels = format_results(predictions, v.metadata.get("thing_classes"))
        
        tracked_objects = mot_tracker.update(boxes_list)
        print(mot_tracker.matched)

        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * v.output.img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * v.output.img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * v.output.img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * v.output.img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            #cls = classes[int(cls_pred)]
            cv2.rectangle(v.output.img, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(v.output.img, (x1, y1-35), (x1+len("CLASS")*19+60, y1), color, -1)
            cv2.putText(v.output.img, "CLASS" + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

        
        #ret = np.asanyarray(outputs['instances'])
        #print(outputs['instances'])
        v.overlay_instances(
            masks=masks,
            boxes=None,
            labels=labels,
            keypoints=None,
            assigned_colors=None,
            alpha=0.3
        )


        speed_time_end = time.time()
        
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

            track = mot_tracker.matched[np.where(mot_tracker.matched[:,0]==i)[0],1]
            print("track: {}".format(track))
            if len(track) > 0:
                track = track[0]
                if i not in mot_tracker.unmatched:
                    try:
                        """
                        if hasattr(mot_tracker.trackers[track], 'distance'):
                            
                            mot_tracker.trackers[track].speed = (mot_tracker.trackers[track].distance - centre_depth)/(speed_time_end - speed_time_start)
                            v.draw_text("{:.2f}m/s".format(mot_tracker.trackers[track].speed), (cX, cY + 40))
                        """
                        if hasattr(mot_tracker.trackers[track], 'position'):
                            #print("From {} to {} in {:.2f}s".format(mot_tracker.trackers[track].distance, centre_depth, speed_time_end - speed_time_start))
                            x1, y1, z1 = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, [cX, cY], centre_depth
                        )
                            

                            mot_tracker.trackers[track].distance_3d = math.sqrt((x1 - mot_tracker.trackers[track].position[0])**2 + (y1 - mot_tracker.trackers[track].position[1])**2 + (z1 - mot_tracker.trackers[track].position[2])**2)
                            mot_tracker.trackers[track].velocity = mot_tracker.trackers[track].distance_3d / (speed_time_end - speed_time_start)

                            v.draw_text("{:.2f}m/s".format(mot_tracker.trackers[track].velocity), (cX, cY + 40))

                            #axins = v.output.ax.inset_axes([cX - 50, cY - 50, 100, 100], transform=v.output.ax.transData)

                            relative_x = (cX - 64) / RESOLUTION_X
                            relative_y = (abs(RESOLUTION_Y - cY) - 36) / RESOLUTION_Y
                            #print("relative_x: {}, relative_y: {}".format(relative_x, relative_y))
                            """
                            128
                            72
                            """
                            ax = v.output.fig.add_axes([relative_x, relative_y, 0.1, 0.1], projection='3d')
                            ax.set_xlim([-2, 2])
                            ax.set_ylim([-2, 2])
                            v_points = [x1 - mot_tracker.trackers[track].position[0], y1 - mot_tracker.trackers[track].position[1], z1 - mot_tracker.trackers[track].position[2]]
                            print("Original position: {}".format(mot_tracker.trackers[track].position))
                            print("New position: {}".format([x1, y1, z1]))
                            print("Differences: {:.2f}, {:.2f}, {:.2f}".format(v_points[0], v_points[1], v_points[2]))
                            #print(v_points)
                            a = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
                            ax.add_artist(a)
                            ax.axis("off")
                            #ax.set_position = ([200, 200, 200, 200])
                            #ax.set_position([500, 500, 500, 500])
                            v.output.fig.add_axes(ax)

                        mot_tracker.trackers[track].distance = centre_depth
                        mot_tracker.trackers[track].position = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, [cX, cY], centre_depth
                        )

                        
                    
                    except IndexError:
                        continue


            v.draw_circle((cX, cY), (0, 0, 0))
            v.draw_text("{:.2f}m".format(centre_depth), (cX, cY + 20))
            
        speed_time_start = time.time()

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #cv2.imshow('Segmented Image', color_image)
        cv2.imshow('Segmented Image', v.output.get_image()[:, :, ::-1])
        #cv2.imshow('Depth', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time_end = time.time()
        total_time = time_end - time_start
        #print("Time to process frame: {:.2f}".format(total_time))
        #print("FPS: {:.2f}\n".format(1/total_time))
        
    pipeline.stop()
    cv2.destroyAllWindows()
    