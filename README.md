# Asymmetric Deep Interaction Network for RGB-D Salient Object Detection

## 1. Overview
### 1.1. Introduction
In recent years, most of the existing RGB-D SOD models use summation or splicing strategies to directly aggregate information from different modalities and decode features from different layers to predict saliency maps. However, they ignore the complementary properties of depth images and RGB images and the effective use of features between the same layers, resulting in a degraded model performance. To address this issue, we propose an asymmetric deep interaction network (ADINet) with three indispensable components with a focus on information fusion \& embedding. Specifically, we design a cross-modal fusion encoder  for enhancing the information fusion \& embedding on semantic signals that is employed to benefit from the mutual interaction of RGB and depth information. Then, we propose a global-and-local feature decoder to enrich the global \& local information for improving the recognition of salient objects. We have conducted the experiments on seven RGB-D benchmarks, and the results demonstrate that the proposed method is superior to or competitive with the state-of-the-art works.

### 1.2. Framework Overview
![alt](https://github.com/yuluofeiyu/ADINet/tree/main/ADINet/imgs/ADINet.png)
### 1.3. Quantitative Results
![alt](https://github.com/yuluofeiyu/ADINet/tree/main/ADINet/imgs/QuantitativeComparisons.png)
### 1.4. PR Curves

### 1.5. Qualitative Results



## Preparing the necessary data
downloading training datasets from [Baidu Drive](https://pan.baidu.com/s/1wfOG3EqyLaM0UH6pwfvpdA)(extraction code: o3o4).\
downloading testing datasets from [Baidu Drive](https://pan.baidu.com/s/1uERpDsb9GIvCACOoCXeJSg)(extraction code: 211k).\
downloading Swin V2 weights ([Swin V2](https://pan.baidu.com/s/1_zZIHiBFOHXZ-F-cJohKTQ)(extraction code: 6hyq)) and move it into [./pretrain/swinv2_base_patch4_window16_256.pth].

## Training
`python train_ADINet.py`

## Testing
`python test_ADINet.py`

## Training and Testing
`python run_ADINet.py`

## Results
We provide saliency maps of ADINet on seven benchmark datasets, including: DUT-RGBD, NJU2K, NLPR, SIP, SSD, LFSD and RedWeb-S from [Baidu Drive](https://pan.baidu.com/s/1c9bv4PbEm_IghfCjF_Y-pw)(extraction code: ADIN).

## Evaluating Results
When training is complete, the predictions for the test set are saved in . /test_maps. We provided [python versions](https://pan.baidu.com/s/1Y1bn4ITcWAOqp-43SNVJbg)(extraction code: dr6d) for evaluation.\
`python main.py`

## Note: Our core code is being organized and will be uploaded later!
