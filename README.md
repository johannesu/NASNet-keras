[![Build Status](https://travis-ci.org/johannesu/NASNet-keras.svg?branch=master)](https://travis-ci.org/johannesu/NASNet-keras)

# NASNetA-Keras

[Keras](https://keras.io/) implementation of NASNet-A. The best performing model from the paper [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) [1].
An extension of [AutoML](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html) [2].

### Reference implementation
Googles' tensorflow-slim implementation: [https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet).


### Example
`NASNet-A (6 @ 768)` from the paper, visualized in tensorboard:

![NASNet-A (6 @ 768)](images/6_768.png)


### References
[1]   __Learning Transferable Architectures for Scalable Image Recognition__.
[https://arxiv.org/abs/1707.07012](https://arxiv.org/abs/1707.07012)
_Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le_.

[2]  __AutoML for large scale image classification and object detection__
[https://research.googleblog.com/2017/11/automl-for-large-scale-image.html](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html)
_Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le_.