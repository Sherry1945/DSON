# Deep Second Order Network

## Introduction
This paper delves deep into the optimization behaviors of deep Global Covariance Pooling (GCP) networks with matrix power normalization. The authors undertake a thorough analysis and understanding of the effect of GCP on deep architectures, primarily from an optimization perspective. The research can be summarized into three successive parts:

Analysis of the impact of GCP with matrix power normalization on deep architectures. This involves examining the behaviors of both optimization loss (e.g., smoother loss landscape and flatter local minima) and gradient computation.
Investigation and improvement of post-normalization key for optimizing GCP networks. This led to the proposal of the DropCov method, a novel technique for efficiently normalizing GCP. The DropCov method was found to be superior or competitive with existing methods in terms of both efficiency and effectiveness.
Exploration of three new benefits of GCP for deep architectures that have not been previously recognized or fully explored.
[Download](https://drive.google.com/file/d/1zVDDmmQWQ-CDDoxjaolkcjI3MACE-rxx/view?usp=drive_link)


## Usage

### Environments
●OS：18.04  
●CUDA：11.6  
●Toolkit：mindspore1.9  
●GPU:GTX 3090 


### Install
●First, Install the driver of NVIDIA  
●Then, Install the driver of CUDA  
●Last, Install cudnn

create virtual enviroment mindspore
conda create -n mindspore python=3.7.5 -y
conda activate mindspore
CUDA 10.1 
```bash
conda install mindspore-gpu cudatoolkit=10.1 -c mindspore -c conda-forge
```
CUDA 11.1 
```bash
conda install mindspore-gpu cudatoolkit=11.1 -c mindspore -c conda-forge
```
validataion 
```bash
python -c "import mindspore;mindspore.run_check()"
```

### Data preparation
Download and extract ImageNet train and val images from http://image-net.org/. (https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:


```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```
### Evaluation
To evaluate a pre-trained model on ImageNet val with GPUs run:

```
CUDA_VISIBLE_DEVICES={device_ids}  python eval.py --data_path={IMAGENET_PATH} --checkpoint_file_path={CHECKPOINT_PATH} --device_target="GPU" --config_path={CONFIG_FILE} &> log &
```

### Training

#### Train with ResNet

You can run the `main.py` to train as follow:

```
mpirun --allow-run-as-root -n {RANK_SIZE} --output-filename log_output --merge-stderr-to-stdout python train.py  --config_path={CONFIG_FILE} --run_distribute=True --device_num={DEVICE_NUM} --device_target="GPU" --data_path={IMAGENET_PATH}  --output_path './output' &> log &
```
For example:

```bash
mpirun --allow-run-as-root -n 4 --output-filename log_output --merge-stderr-to-stdout python train.py  --config_path="./config/resnet50_imagenet2012_config.yaml" --run_distribute=True --device_num=4 --device_target="GPU" --data_path=./imagenet --output_path './output' &> log &
```



## Our Works

|Works         | Paper | Code|                                                         
| ------------------ | ----- | ------- | 
| Towards a Deeper Understanding of Global Covariance Pooling in Deep Learning: An Optimization Perspective  |  [Link](https://ieeexplore.ieee.org/document/10269023)|[Link](https://github.com/Terror03/GCP-DropCov/blob/main/README.md)   |
| DropCov: A Simple yet Effective Method for Improving Deep Architectures   | [Link](https://papers.nips.cc/paper_files/paper/2022/hash/d9888cc7baa04c2e44e8115588133515-Abstract-Conference.html)  |   19.6  |
| ResNet-18+ISqrt(Ours)   | 72.25  |   19.6  | 
| ResNet-34   |  73.68 |  21.8   |   3.66  |   


## References
[GCP_CVPR2020]Wang, Qilong, Li Zhang, Banggu Wu, Dongwei Ren, Peihua Li, Wangmeng Zuo, and Qinghua Hu. "What deep CNNs benefit from global covariance pooling: An optimization perspective." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition(CVPR), pp. 10771-10780. 2020.

[DropCov_NIPS2022]Wang, Qilong, Mingze Gao, Zhaolin Zhang, Jiangtao Xie, Peihua Li, and Qinghua Hu. "DropCov: a simple yet effective method for improving deep architectures." Advances in Neural Information Processing Systems 35 (NIPS): 33576-33588. 2022.

[ISqrt_CVPR2018]Li, Peihua, Jiangtao Xie, Qilong Wang, and Zilin Gao. "Towards faster training of global covariance pooling networks by iterative matrix square root normalization." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition(CVPR), pp. 947-955. 2018.
