# Asymmetric Deep Interaction Network for RGB-D Salient Object Detection



### Preparing the necessary data
downloading training datasets from [Baidu Drive](https://pan.baidu.com/s/1wfOG3EqyLaM0UH6pwfvpdA)(extraction code: o3o4).\
downloading testing datasets from [Baidu Drive](https://pan.baidu.com/s/1uERpDsb9GIvCACOoCXeJSg)(extraction code: 211k).\
downloading Swin V2 weights ([Swin V2](https://pan.baidu.com/s/1_zZIHiBFOHXZ-F-cJohKTQ)(extraction code: 6hyq)) and move it into [./pretrain/swinv2_base_patch4_window16_256.pth].

### Training
python train_ADINet.py

### Testing
python test_ADINet.py

### Training and Testing
python run_ADINet.py

### Results
We provide saliency maps of ADINet on seven benchmark datasets, including: DUT-RGBD, NJU2K, NLPR, SIP, SSD, LFSD and RedWeb-S from [Baidu Drive](https://pan.baidu.com/s/1c9bv4PbEm_IghfCjF_Y-pw)(extraction code: ADIN).

### Evaluating Results
When training is complete, the predictions for the test set are saved in . /test_maps. We provided [python versions](https://pan.baidu.com/s/1Y1bn4ITcWAOqp-43SNVJbg)(extraction code: dr6d) for evaluation.\
python main.py

### Note: Our core code is being organized and will be uploaded later!
