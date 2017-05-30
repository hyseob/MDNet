## MDNet: Multi-Domain Convolutional Neural Network Tracker

Created by [Hyeonseob Nam](https://kr.linkedin.com/in/hyeonseob-nam/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at POSTECH

Project Webpage: http://cvlab.postech.ac.kr/research/mdnet/


### News
**(May 28, 2017) Python implementation of MDNet is avaliable! [[py-MDNet]](https://github.com/HyeonseobNam/py-MDNet)**



### Introduction

MDNet is the state-of-the-art visual tracker based on a CNN trained on a large set of tracking sequences,
and the winner tracker of [The VOT2015 Challenge](http://www.votchallenge.net/vot2015/).

Detailed description of the system is provided by our [paper](http://arxiv.org/abs/1510.07945).

This software is implemented using [MatConvNet](http://www.vlfeat.org/matconvnet/) and part of [R-CNN](https://github.com/rbgirshick/rcnn).

### Citation

If you're using this code in a publication, please cite our paper.

	@InProceedings{nam2016mdnet,
	author = {Nam, Hyeonseob and Han, Bohyung},
	title = {Learning Multi-Domain Convolutional Neural Networks for Visual Tracking},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2016}
	}


### License

This software is being made available for research purpose only.
Check LICENSE file for details.


### System Requirements

This code is tested on 64 bit Linux (Ubuntu 14.04 LTS).

**Prerequisites** 
  0. MATLAB (tested with R2014a)
  0. MatConvNet (tested with version 1.0-beta10, included in this repository)
  0. For GPU support, a GPU (~2GB memory) and CUDA toolkit according to the [MatConvNet installation guideline](http://www.vlfeat.org/matconvnet/install/) will be needed.


### Installation

  0. Compile MatConvNet according to the [installation guideline](http://www.vlfeat.org/matconvnet/install/). An example script is provided in 'compile_matconvnet.m'.
  0. Run 'setup_mdnet.m' to set the environment for running MDNet.


### Online Tracking using MDNet

**Pretrained Models**

If you only need to run the tracker, you can use the pretrained MDNet models:
  0. models/mdnet_vot-otb.mat (trained on VOT13,14,15 excluding OTB)
  0. models/mdnet_otb-vot14.mat (trained on OTB excluding VOT14)
  0. models/mdnet_otb-vot15.mat (trained on OTB excluding VOT15)

**Demo**
  0. Run 'tracking/demo_tracking.m'.

The demo performs online tracking on *'Diving'* sequence using a pretrained model 'models/mdnet_vot-otb.mat'.

In case of out of GPU memory, decrease *opts.batchSize_test* in 'tracking/mdnet_init.m'.
You can also disable the GPU support by setting *opts.useGpu* in 'tracking/mdnet_init.m' to false (not recommended).


### Learning MDNet
  
**Preparing Datasets**

You may need OTB and VOT datasets for learning MDNet models. You can also use other datasets by configuring 'utils/genConfig.m'.
  0. Download [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html) and [VOT](http://www.votchallenge.net/) datasets.
  0. Locate the OTB sequences in 'dataset/OTB' and VOT201x sequences in 'dataset/VOT/201x', or modify the variables *benchmarkSeqHome* in 'utils/genConfig.m' properly.

**Demo**
  0. Run 'pretraining/demo_pretraining.m'.

The demo trains new MDNet models using OTB or VOT sequences.
