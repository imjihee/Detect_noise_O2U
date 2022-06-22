### O2U-Net
paper "O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks" code

```shell
noise detection run：python main.py  --network=resnet101 --transforms=true
```


---
#### jihee ver
***

2022/05/09
1. Remove Caffe2 thread-pool leak warning: [link](https://github.com/pytorch/pytorch/commit/567e6d3a8766133f384eb1e00635b21ed638d187)

2. *10 epoch experiment result*

epoch:10 lr:0.009100 train_loss: 3.1395677614974975 test_accuarcy:55.330000 noise_accuracy:0.840500 top 0.1 noise accuracy:0.982200

3. model: ResNet50, ResNet101

2022/05/26
1. python main.py  --network=resnet101 --transforms=true (network: resnet50 / resnet101)

2022/6/3
1. first stage, second stage: *main.py*
2. third stage: *curriculum.py* : apply curriculum learning with masked dataset