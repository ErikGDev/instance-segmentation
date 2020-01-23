<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

This is a fork of [Facebook AI Research's](https://github.com/facebookresearch) implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870), Detectron2. Detectron2 is a complete write-up from its previous version
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/). This Mask R-CNN implementation is powered by [PyTorch](https://pytorch.org) and is based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

In this project, real-time video and depth values from a [Intel® RealSense™ D435 camera](https://www.intelrealsense.com/depth-camera-d435/) are inputted into Detectron2's Mask R-CNN model. The output is the same real-time video (3-6fps) with instance segmentation masks and labels superimposed. The median depth values of each object are also outputted. 

<div>
  <img src="images/gif.gif"/>
</div>

## Usage

**Requirements/Dependencies**

- Linux or macOS
- Python ≥ 3.6
- PyTorch ≥ 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
  Please ensure that your version of CUDA is also compatible when installing.
- OpenCV `pip install opencv-python`
- PyRealSense `pip install pyrealsense2`
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- gcc & g++ ≥ 4.9

**Installation**

For the installation of Detectron2 and its dependencies, please refer to the [official Detectron2 GitHub](https://github.com/facebookresearch/detectron2)

**After Installation**

* Copy and paste main.py from this directory into your new Detectron2 directory.
* To perform instance segmentation straight from a D435 camera attached to a USB port:
  * Access the Detectron2 directory and type 'python3 main.py'. 
* If using .bag files:
  * Type 'python3 main.py --file={filename}' where {filename} is the name of the input .bag file. To create .bag files, use d435_to_file.py in the tools folder.

## Accuracy and Specifications of Model

**Segmentation Validation Results**

|  | Backbone | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| :--- | :--- | :---: | :---: | :---: |  :---:  | :---: | :---: |
| Original Mask R-CNN   | ResNet-101-FPN  | 35.7 | 58.0 | 37.8 | 15.5 | 38.1 | 52.4 |
| Matterport Mask R-CNN | ReSNet-101-FPN | 38.0 | 55.8 | <b>41.3</b> | 17.9 | <b>45.5</b> | <b>55.9</b> |
| Detectron2 Mask R-CNN | ReSNet-101-FPN | <b>38.6</b> | <b>60.4</b> | <b>41.3</b> | <b>19.5</b> | 41.3 | 55.3 |

Validation tests were perfomed on the segmentation masks created on the 2017 COCO validation dataset. The standard COCO validation metrics include average AP over IoU thresholds, AP<sub>50</sub>, AP<sub>75</sub>, and AP<sub>S</sub>, AP<sub>M</sub> and AP<sub>L</sub> (AP at different scales). These results were then compared to COCO validation results from the [original paper](https://arxiv.org/abs/1703.06870) and a popular [Mask R-CNN implementation by Matterport](https://github.com/matterport/Mask_RCNN). Clearly, Detectron2's Mask R-CNN outperforms the original Mask R-CNN and Matterport's Mask R-CNN with respect to average precision. It also outperformed state-of-the art COCO segmentation competition winners from the [2015 and 2016 challenge](http://cocodataset.org/#detection-leaderboard). The reason the competition winners from 2017 and 2018 were not chosen, was to avoid overfitting. Furthermore these models trade a slower inference time for a higher accuracy.

**Why this model?**

Detectron2's Mask R-CNN with a ReSNet-101-FPN backbone was determined to be the optimal model. Upon comparing Detectron2 to [MMDetection's models](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md), which won first place in the [2018 segmentation COCO challenge](http://cocodataset.org/#detection-leaderboard), it is evident that the choice of model is appropriate for high-speed real-time video. 

When comparing [Detectron2's Mask R-CNN](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn) to [MMDetection's Mask R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md#mask-r-cnn), Detectron2 outperforms in both mask AP (38.6 vs 35.9) and inference time (0.070 s/im vs 0.105 s/im). MMDetectron does have models that are slightly more accurate than Detectron2's Mask R-CNN implementation, such as [the Hybrid Task Cascade model (HTC)](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md#hybrid-task-cascade-htc) however these often result in models that output masks at less than 4 fps. When adding the time to ouput the superimposed images, this would be insufficient for real-time.

Detectron2's Model Zoo displays the inference time and Mask AP for each model provided. For the Mask R-CNN models, the FPN model with a ResNet101 backbone has the best Mask AP for the short time it takes for inferences.

<img src="images/detectron2_model_zoo.png" />

## Intel® RealSense™ D435 Camera

According to Intel's paper, [Best-Known-Methods for Tuning Intel® RealSense™ D400 Depth Cameras for Best Performance](https://www.intelrealsense.com/wp-content/uploads/2019/11/BKMs_Tuning_RealSense_D4xx_Cam.pdf), The depth RMS (root mean square) error increases rapidly when placing objects further away, especially when the distance is greater than 3m. The orange line on the graph below represents the depth RMS error on a D435 with HFOV=90deg, Xres=848, baseline=50mm and for subpixel=0.08.

<img src="images/d435_rms_error.png" />

**Insert My Testing Here**

Testing was performed on this program, where the real distances of objects from the D435 were compared to the distance measured by the stereo sensors on the D435. Rather than the depth RMS error, the absolute depth error was compared to the real distance of the object to the D435.

This graph shows that the absolute error exponentially increases when the distance increases. This means the depth recordings will be most accurate when the object is closer to the camera.

**When does the Object Record 0m?**

When the object is too close to the camera, the depth values will return 0m. This threshold is known as MinZ. The formula for calculating MinZ is

MinZ(mm) = focal length(pixels)𝒙 Baseline(mm)/126

Therefore with a depth resolution of 848x480, the MinZ is ~16.8cm. If the object is within this distance, no value is returned.

Similar to MinZ, MaxZ exists too. For the D435, the MaxZ is [approximately 10m](https://ark.intel.com/content/www/us/en/ark/products/128255/intel-realsense-depth-camera-d435.html). Any object outside this range will also be recorded as 0m.

Sometimes objects can be recorded as 0m even though they are inside the MinZ and MaxZ threshold. This usually occurs when there is too much noise on the depth image. This can occur when the target is not well textured. For more information on how to configure the D435 for specific environments and objects, refer to [this paper](https://www.intelrealsense.com/wp-content/uploads/2019/11/BKMs_Tuning_RealSense_D4xx_Cam.pdf).

**How is each Depth Value Calculated?**

To find the distance of each object, the median depth pixel is used. All pixels associated to the object are abstracted to a histogram with a max distance of 10m (Max range of the D435), and 500 bins. This means that the depth values will change with intervals of 0.02m.

For smaller intervals of 0.01m, change the NUM_BINS constant to 1000, and change 

```centre_depth = "{:.2f}m".format(x / 50)```
to
```centre_depth = "{:.2f}m".format(x / 100)```
