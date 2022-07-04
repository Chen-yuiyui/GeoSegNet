# Investigate Indistinguishable Points in Semantic Segmentation of 3D Point Cloud







## Introduction

This repository propose python scripts for point cloud semantic segmentation. The library is coded with PyTorch.

The conference paper is here:
https://arxiv.org/pdf/2103.10339.pdf?ref=https://githubhelp.com




## Citation

If you use this code in your research, please consider citing:
(citation will be updated as soon as 3DOR proceedings will be released)

```
@inproceedings{xu2021investigate,
  title={Investigate Indistinguishable Points in Semantic Segmentation of 3D Point Cloud},
  author={Xu, Mingye and Zhou, Zhipeng and Zhang, Junhao and Qiao, Yu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3047--3055},
  year={2021}
}
```


## Data composition
The data is placed under the ```./data/s3dis_data ``` directory, as follows
```./data/s3dis_data/Area_1/conferenceRoom_1/xyzrgb.npy ```



## Platform

The code was tested on Ubuntu 16.04 with Anaconda.

## Dependencies

- Pytorch
- Scikit-learn for confusion matrix computation, and efficient neighbors search  
- TQDM for progress bars
- PlyFile
- H5py

All these dependencies can be install via conda in an Anaconda environment or via pip.


### Nearest neighbor module

The ```nearest_neighbors``` directory contains a very small wrapper for [NanoFLANN](https://github.com/jlblancoc/nanoflann) with OpenMP.
To compile the module:
```
cd nearest_neighbors
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
