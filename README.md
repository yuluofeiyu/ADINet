# Asymmetric Deep Interaction Network for RGB-D Salient Object Detection


### Preparing the necessary data
downloading training datasets from [Baidu Drive](https://pan.baidu.com/s/13MLjRuF5JpRGxJwm8qU0xQ)(extraction code: 5lg3).\
downloading testing datasets from [Baidu Drive](https://pan.baidu.com/s/1uERpDsb9GIvCACOoCXeJSg)(extraction code: 211k).\
downloading Swin V2 weights ([Swin V2](https://pan.baidu.com/s/1_zZIHiBFOHXZ-F-cJohKTQ)(extraction code: 6hyq)) and move it into [./pre/swinv2_base_patch4_window16_256.pth].


### Training
train_ADINet.py

### Testing
test_ADINet.py

### Training and # Testing
python run_ADINet.py

### Results
We provide saliency maps of ADINet on seven benchmark datasets, including: DUT-RGBD, NJU2K, NLPR, SIP, SSD, LFSD and RedWeb-S from [Baidu Drive](https://pan.baidu.com/s/1c9bv4PbEm_IghfCjF_Y-pw)(extraction code: ADIN).




