# GeoSegNet: Point Cloud Semantic Segmentation via Geometric Encoder-Decoder Modeling


It has been published online in The Visual Computer:(https://link.springer.com/article/10.1007/s00371-023-02853-7))




## Introduction

This repository propose python scripts for point cloud semantic segmentation. The library is coded with PyTorch.


## Data composition
The data is placed under the ```./data/s3dis_data ``` directory, as follows
```./data/s3dis_data/Area_1/conferenceRoom_1/xyzrgb.npy ```



## Platform

The code was tested on Ubuntu 20.04 with Anaconda, pytorch3.6, CUDA11.1.

## Dependencies

* Install dependencies
 ```pip install -r requirements.txt```


Building only the CUDA kernels
----------------------------------
```
  pip install pointnet2_ops_lib/.
```
    # Or if you would like to install them directly (this can also be used in a requirements.txt)



### Nearest neighbor module

The ```nearest_neighbors``` directory contains a very small wrapper for [NanoFLANN](https://github.com/jlblancoc/nanoflann) with OpenMP.
To compile the module:
```
cd \convpoint\knn
python setup.py install --home="."
```


## Data preparation

Data is prepared using the ```./excample/s3dis/prepare_s3dis_label.py```.



### Training

```
cd ./s3dis
```

For training on area 5:

```
python s3dis_seg.py --rootdir path_to_data_processed/ --area 5 --savedir path_to_save_directory
```



### Testing

For testing on area 5:
```
python s3dis_seg.py --rootdir path_to_data_processed --area 5 --savedir path_to_save_directory --test
```


## Evaluation

```
python s3dis_eval.py --datafolder path_to_data_processed --predfolder pathèto_model --area 5
```



## Acknowledgement
We include the following PyTorch 3rd-party libraries:


[1] [ConvPoint] (https://github.com/aboulch/ConvPoint)

[2] [GSNet] (https://github.com/MingyeXu/GS-Net)

[3] [IAF-Net] (https://github.com/MingyeXu/IAF-Net)

[4] [SCF-Net] (https://github.com/leofansq/SCF-Net)

[5] [RandLA-Net] (https://github.com/QingyongHu/RandLA-Net)

[6] [PointMLP] (https://github.com/ma-xu/pointMLP-pytorch)
