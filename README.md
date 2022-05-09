## O2U-Net
paper "O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks" code

'''shell
noise detection run：sudo python main.py  --network=resnet101 --transforms=true
'''

---
jihee ver
---
2022/05/09
1. Remove Caffe2 thread-pool leak warning
[link](https://github.com/pytorch/pytorch/commit/567e6d3a8766133f384eb1e00635b21ed638d187)

2. 10 epoch experiment result
epoch:10 lr:0.009100 train_loss: 3.1395677614974975 test_accuarcy:55.330000 noise_accuracy:0.840500 top 0.1 noise accuracy:0.982200

