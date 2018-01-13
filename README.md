[![Build Status](https://travis-ci.org/johannesu/NASNet-keras.svg?branch=master)](https://travis-ci.org/johannesu/NASNet-keras)

# NASNetA-Keras

[Keras](https://keras.io/) implementation of NASNet-A. The best performing model from the paper [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) [1].
An extension of [AutoML](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html) [2].

## Install
System requirements on Ubuntu 16.04

```bash
sudo apt-get install python3-pip python3-tk
```

Install python requirements and the package
```bash
pip3 install https://github.com/johannesu/NASNet-keras/archive/master.zip
```

You can now use the package in python

```python
from nasnet import mobile
model = mobile()
model.summary()
```

## Demo

See [demo.ipynb](demo.ipynb).

### Reference implementation
Googles' tensorflow-slim implementation: [https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet).


### Pretrained weights
Models trained with the reference implementation can be convert to this model.
This includes the two trained models provided by Google [https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet).

* Setup models, download tensorflow checkpoints and convert them to Keras.

```python
import nasnet

# NASNet-A_Mobile_224
model = nasnet.mobile(weights='imagenet')

# NASNet-A_Large_331
model = nasnet.large(weights='imagenet')
```

Converting the checkpoints can take a few minutes, the work is cached and will be fast the second call.


### Model visualization
`NASNet-A (6 @ 768)` from the paper, visualized in tensorboard:

![NASNet-A (6 @ 768)](images/6_768.png)


### References
[1]   __Learning Transferable Architectures for Scalable Image Recognition__.
[https://arxiv.org/abs/1707.07012](https://arxiv.org/abs/1707.07012)
_Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le_.

[2]  __AutoML for large scale image classification and object detection__
[https://research.googleblog.com/2017/11/automl-for-large-scale-image.html](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html)
_Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le_.