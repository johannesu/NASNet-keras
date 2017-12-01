[![Build Status](https://travis-ci.org/johannesu/NASNet-keras.svg?branch=master)](https://travis-ci.org/johannesu/NASNet-keras)

# NASNetA-Keras

[Keras](https://keras.io/) implementation of NASNet-A. The best performing model from the paper [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) [1].
An extension of [AutoML](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html) [2].

## Getting started

Look at [demo.ipynb](demo.ipynb).

## Reference implementation
Googles' tensorflow-slim implementation: [https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet).


## Pretrained weights
Weights  trained with the reference implementation can be convert to this model.
This includes the two pretrained models provided by Google [https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet).

* Setup models, download pretrained weights and save weights to Keras compatible .h5 file:

```python
import nasnet

# NASNet-A_Mobile_224
model = nasnet.mobile(load_weights=True)
model.save_weights('mobile.h5')

# NASNet-A_Large_331
model = nasnet.large(load_weights=True)
model.save_weights('large.h5')
```

* Load Keras weights:

```python
import nasnet

model = nasnet.mobile()
model.load_weights('mobile.h5')
```

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