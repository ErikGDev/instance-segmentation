<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

This is a fork of [Facebook AI Research's](https://github.com/facebookresearch) implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870), Detectron2. Detectron2 is a complete write-up from its previous version
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/). This Mask R-CNN implementation is powered by [PyTorch](https://pytorch.org).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

## Installation

For the installation of Detectron2, please refer to the [official Detectron2 GitHub](https://github.com/facebookresearch/detectron2)

## Accuracy of Model

**Intel® RealSense™ D435 Camera**

The depth value uncertaincy for the D435 camera is less than one percent of the distance from the object. If the camera is 1 metre from the object, the expected uncertainty is between 2.5mm and 5mm.
When the object is too close to the camera, the depth values will return 0mm. This threshold is known as MinZ. The formula for calculating MinZ is

MinZ(mm) = focal length(pixels)𝒙 Baseline(mm)/126

Therefore with a depth resolution of 848x480, the MinZ is ~16.8cm. If the object is within this distance, no value is returned.

**Segmentation Validation Results**

|  | Backbone | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| :--- | :--- | :---: | :---: | :---: |  :---:  | :---: | :---: |
| Original Mask R-CNN   | ResNet-101-FPN  | 35.7 | 58.0 | 37.8 | 15.5 | 38.1 | 52.4 |
| Matterport Mask R-CNN | ReSNet-101-FPN | 38.0 | 55.8 | <b>41.3</b> | 17.9 | <b>45.5</b> | <b>55.9</b> |
| Detectron2 Mask R-CNN | ReSNet-101-FPN | <b>38.6</b> | <b>60.4</b> | <b>41.3</b> | <b>19.5</b> | 41.3 | 55.3 |

I performed validation tests on the segmentation masks created on the 2017 COCO validation dataset. The standard COCO validation metrics include average AP over IoU thresholds, AP<sub>50</sub>, AP<sub>75</sub>, and AP<sub>S</sub>, AP<sub>M</sub> and AP<sub>L</sub> (AP at different scales). These results were then compared to COCO validation results from the [original paper](https://arxiv.org/abs/1703.06870). Clearly, Matterport's Mask R-CNN outperforms the original Mask R-CNN with respect to average precision. It also outperformed state-of-the art COCO segmentation competition winners from the 2015 and 2016 challenge. The reason I chose not to use the models of competition winners from 2017 and 2018 is to avoid overfitting.

## Citing Detectron

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
