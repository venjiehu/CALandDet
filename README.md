# CAlandDet model for Landslide Mapping

## Requestments

To set up the required environment, the 3 steps is need:

1. Check you CUDA vision and install **pytorch** and **torchvision** from [the official website](https://pytorch.org/)；

- PyTorch and torchvision 2.1.0+cu121 is recommended version.

2. Install **GDAL** using conda, please run:

    ```bash
    conda install GDAL==3.6.2
    ```

3. To install other required package, please run:

    ```bash
    pip install -r requirements.txt
    ```


## CAlandDet

The model was originally named ZiHao Net, derived from my friend's name, but this name seemed insufficiently academic and too casual for paper, so it was changed to CAlanDet. However, you might still find traces of the previous name in the code files.

## Baseline dataset
This dataset [(for google drive)](https://drive.google.com/file/d/1YpIN08Ou2w3GJLVgq-Ts7Xf4UV6fGV8v/view?usp=sharing) is based on:<br> 
- [Bijie Landslide Dataset](https://gpcv.whu.edu.cn/data/Bijie_pages.html);<br> 
- [High-precision aerial imagery and interpretation dataset of landslide and debris flow disaster in Sichuan and surrounding areas](https://www.scidb.cn/en/detail?dataSetId=803952485596135424#p2);<br> 
- [The Global Very-High-Resolution Landslide Mapping Dataset](https://github.com/zxk688/GVLM).<br>

We appreciate their selfless contributions to academic research. For more details, please refer to the original articles listed below:<br> 


1. <small><em> Ji, S.; Yu, D.; Shen, C.; Li, W.; Xu, Q. Landslide Detection from an Open Satellite Imagery and Digital Elevation Model Dataset Using Attention Boosted Convolutional Neural Networks. Landslides 2020, 17, 1337–1352.</em></small>
2. <small><em> Zhang, X.; Yu, W.; Pun, M.-O.; Shi, W. Cross-Domain Landslide Mapping from Large-Scale Remote Sensing Images Using Prototype-Guided Domain-Aware Progressive Representation Learning. ISPRS J. Photogramm. Remote Sens. 2023, 197, 1–17.</em></small>
3. <small><em> Zeng, C.; Cao, Z.; Su, F.; Zeng, Z.; Yu, C. A Dataset of High-Precision Aerial Imagery and Interpretation of Landslide and Debris Flow Disaster in Sichuan and Surrounding Areas between 2008 and 2020. China Sci. Data 2022, 7, 195–205.</em></small>

## Large scale image inference tool

tif2mask.py is a small tool I developed for generating predictions on large scale imagery. Its segmentation strategy intentionally discards border regions that lack sufficient context, resulting in more cohesive segmentation results. If you need to process large scale segmentation outputs, this tool may be useful.
