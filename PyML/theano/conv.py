import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy as np
import pylab
from PIL import Image


'''
+  a 4D tensor corresponding to a mini-batch of input images. The shape of the
   tensor is as follows: [mini-batch size, number of input feature maps, image
   height, image width].

+  a 4D tensor corresponding to the weight matrix W. The shape of the tensor is:
   [number of feature maps at layer m, number of feature maps at layer m-1, filter
   height, filter width]
'''
rng = np.random.RandomState(1234)

input = T.tensor4(name='input')

w_shp = (2, 3, 9, 9)
w_bound = np.sqrt(3 * 9 * 9)
W = theano.shared(np.asarray(
    rng.uniform(
        low=-1.0 / w_bound,
        high=1.0 / w_bound,
        size=w_shp),
    dtype=input.dtype), name='W'
)

b_shp = (2,)
b = theano.shared(np.asarray(
    rng.uniform(low=-.5, high=.5, size=b_shp),
    dtype=input.dtype), name='b'
)

conv_out = conv2d(input, W)
conv = theano.function([input], conv_out)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)

wolf = Image.open(open('3wolfmoon.jpg'))

# dimensions are (height, width, channel)
img = np.asarray(wolf, dtype='float64') / 256

wolf_4t = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
con_op = conv(wolf_4t)
print con_op.shape

filtered_img = f(wolf_4t)

pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()

pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])

pylab.show()
