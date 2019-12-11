# An exntension of maskrcnn benchmark on object detection

This repository contains some extension works on the original maskrcnn benchmark repo. Below is a list of modules added.

1. FCOS (https://arxiv.org/abs/1904.01355)
2. cosine anealing learning step with warm up (https://arxiv.org/abs/1812.01187)
3. squeeze and excitation net backbone (https://arxiv.org/abs/1709.01507)
4. CBAM module backbone (https://arxiv.org/abs/1807.06521)
5. LIBRA RCNN, with balanced IoU sampling, balanced feature map and balanced L1 loss implemented (https://arxiv.org/pdf/1904.02701.pdf)
6. path aggregation neck (https://arxiv.org/pdf/1803.01534.pdf) PS: I don't add the xconv heavy head in this. The adaptive RoI pooling code doesn't work correctly in xconv. Only 2 MLP head with adaptive RoI pooling is implemented.   
7. cascade rcnn head (https://arxiv.org/abs/1712.00726)
8. Global context block module in ResNet (https://arxiv.org/abs/1904.11492)
9. hrnet backbone (https://arxiv.org/abs/1904.04514)
10. negative sample training 
11. OHEM (in process)

If you have any questions, please start an issue. I will try my best to answer them. The training/inferencing config files should be more helpful with digging into codes. Upon request, I can upload some backbone models which are pre-trained on ImageNet.

