# P2D

Predict to Detect: Prediction-guided 3D Object Detection using Sequential Images, ICCV 2023 ([Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Predict_to_Detect_Prediction-guided_3D_Object_Detection_using_Sequential_Images_ICCV_2023_paper.pdf), [Supplementary](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Kim_Predict_to_Detect_ICCV_2023_supplemental.pdf))

### Abstract
Recent camera-based 3D object detection methods have introduced sequential frames to improve the detection performance hoping that multiple frames would mitigate the large depth estimation error.
Despite improved detection performance, prior works rely on naive fusion methods (e.g., concatenation) or are limited to static scenes (e.g., temporal stereo), neglecting the importance of the motion cue of objects. These approaches do not fully exploit the potential of sequential images and show limited performance improvements. To address this limitation, we propose a novel 3D object detection model, P2D (Predict to Detect), that integrates a prediction scheme into a detection framework to explicitly extract and leverage motion features. P2D predicts object information in the current frame using solely past frames to learn temporal motion features. We then introduce a novel temporal feature aggregation method that attentively exploits Bird's-Eye-View (BEV) features based on predicted object information, resulting in accurate 3D object detection. Experimental results demonstrate that P2D improves mAP and NDS by 3.0\% and 3.7\% compared to the sequential image-based baseline, proving that incorporating a prediction scheme can significantly improve detection accuracy. 


<img src="figs/architecture.png" width="1000">

# Getting Started
- [Installation](docs/install.md)
- [Data preparation](docs/data_preparation.md)
- [Run](docs/run.md)


# Model Zoo
|Model | Backbone | Weight| mAP | NDS|
| - | - | - | -| -|
| P2D| ResNet50 |[link](https://drive.google.com/file/d/1Cj6Dwvs6hS6iUKhZEpPfScs8t7Eh4_9G/view?usp=sharing) | 36.0 | 47.4 |
| P2D| ConvNext-B | [link](https://drive.google.com/file/d/1r_dCbGEQX4HmABag8EuET6J1hnUMIONX/view?usp=sharing) | 46.0 | 55.1 |
# Citation
```
@inproceedings{kim2023predict,
  title={Predict to Detect: Prediction-guided 3D Object Detection using Sequential Images},
  author={Sanmin Kim, Youngseok Kim, In-Jae Lee, and Dongsuk Kum},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18057--18066},
  year={2023}
}
```
# Acknowledgement
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
