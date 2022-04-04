from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten, Multiply, Lambda, Add
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu, sigmoid
from keras.initializers import RandomNormal
import os
import keras.backend as K


K.set_image_data_format('channels_last')
channel_axis=-1
channel_first = False


def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a)  # for convolution kernel
    k.conv_weight = True
    return k


conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)  # for batch normalization


# In[5]:

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer=conv_init, *a, **k)


def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                              gamma_initializer=gamma_init)


def BASIC_D(nc_in, ndf, max_layers=3, use_sigmoid=True):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """
    if channel_first:
        input_a = Input(shape=(nc_in, None, None))
    else:
        input_a = Input(shape=(None, None, nc_in))
    _ = input_a
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name='First')(_)
    _ = LeakyReLU(alpha=0.2)(_)

    for layer in range(1, max_layers):
        out_feat = ndf * min(2 ** layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same",
                   use_bias=False, name='pyramid.{0}'.format(layer)
                   )(_)
        _ = batchnorm()(_, training=1)
        _ = LeakyReLU(alpha=0.2)(_)

    out_feat = ndf * min(2 ** max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4, use_bias=False, name='pyramid_last')(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name='final'.format(out_feat, 1),
               activation="sigmoid" if use_sigmoid else None)(_)
    return Model(inputs=[input_a], outputs=_)




def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    max_nf = 8 * ngf

    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        assert s >= 2 and s % 2 == 0
        if nf_next is None:
            nf_next = min(nf_in * 2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                   padding="same", name='conv_{0}'.format(s))(x)
        if s > 2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s // 2, nf_next)
            x = Concatenate(axis=channel_axis)([x, x2])
        x = Activation("relu")(x)
        # TODO: add attention of phase here

        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer=conv_init,
                            name='convt.{0}'.format(s))(x)
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <= 8:
            x = Dropout(0.5)(x, training=1)
        return x

    s = isize if fixed_input_size else None
    if channel_first:
        _ = inputs = Input(shape=(nc_in, s, s))
    else:
        _ = inputs = Input(shape=(s, s, nc_in))
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = Activation('tanh')(_)
    return Model(inputs=inputs, outputs=[_])



def UNET_G_att(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    use_batchnorm = False
    s = isize
    nf_in = ngf
    print('in size', s,s,nc_in)
    x = inputs = Input(shape=(s, s, nc_in))
    #s = 512
    xc = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x)
    x1 = LeakyReLU(alpha=0.2)(xc)
    # s = 256, nf = 128
    s = s // 2
    nf_in = nf_in * 2
    x1c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x1)
    x2 = LeakyReLU(alpha=0.2)(x1c)
    # s = 128 nf = 256
    s = s // 2
    nf_in = nf_in * 2
    x2c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x2)
    x3 = LeakyReLU(alpha=0.2)(x2c)
    # s = 64 nf = 512
    s = s // 2
    nf_in = nf_in * 2
    x3c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x3)
    x4 = LeakyReLU(alpha=0.2)(x3c)
    # s = 32 nf = 512
    s = s // 2
    nf_in = nf_in
    x4c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x4)
    x5 = LeakyReLU(alpha=0.2)(x4c)
    # s = 16 nf = 512
    s = s // 2
    nf_in = nf_in
    x5c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x5)

    x6 = Activation("relu")(x5c)

    x6 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x6)
    x6 = Cropping2D(1)(x6)
    x4 = Concatenate(axis=channel_axis)([x6, x4c])
    x4 = Activation("relu")(x4)

    # 32, 256
    s = s * 2
    nf_in = nf_in // 2
    x4 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x4)
    x4 = Cropping2D(1)(x4)
    x3 = Concatenate(axis=channel_axis)([x4, x3c])
    x3 = Activation("relu")(x3)

    # 64, 128
    s = s * 2
    nf_in = nf_in // 2
    x3 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x3)
    x3 = Cropping2D(1)(x3)
    x2 = Concatenate(axis=channel_axis)([x3, x2c])
    x2 = Activation("relu")(x2)

    # 128, 64
    s = s * 2
    nf_in = nf_in // 2
    x2 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x2)
    x2 = Cropping2D(1)(x2)
    x1 = Concatenate(axis=channel_axis)([x2, x1c])
    x1 = Activation("relu")(x1)

    # 256, 32
    s = s * 2
    nf_in = nf_in // 2
    x1 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x1)
    x1 = Cropping2D(1)(x1)
    x = Concatenate(axis=channel_axis)([x1, xc])
    x = Activation("relu")(x)

    # 512,16
    s = s * 2
    x = Conv2DTranspose(nc_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x)
    x = Cropping2D(1)(x)

    x = Activation('tanh')(x)

    return Model(inputs=inputs, outputs=[x])


def UNET_G_phaseatt(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    use_batchnorm = False
    s = s_p = isize
    nf_in = nf_in_p = ngf
    print('in size', s,s,nc_in)
    x = inputs = Input(shape=(s, s, nc_in))
    intens_x = crop(3,1,2)(x)
    phase_x = crop(3,2,3)(x)
    #s = 512
    xc = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(intens_x)
    x1 = LeakyReLU(alpha=0.2)(xc)
    ## intensity branch
    # s = 256, nf = 128
    s = s // 2
    nf_in = nf_in * 2
    x1c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x1)
    x2 = LeakyReLU(alpha=0.2)(x1c)
    # s = 128 nf = 256
    s = s // 2
    nf_in = nf_in * 2
    x2c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x2)
    x3 = LeakyReLU(alpha=0.2)(x2c)
    # s = 64 nf = 512
    s = s // 2
    nf_in = nf_in * 2
    x3c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x3)
    x4 = LeakyReLU(alpha=0.2)(x3c)
    # s = 32 nf = 512
    s = s // 2
    nf_in = nf_in
    x4c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x4)
    x5 = LeakyReLU(alpha=0.2)(x4c)
    # s = 16 nf = 512
    s = s // 2
    nf_in = nf_in
    x5c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
               padding="same", name='conv_{0}'.format(s))(x5)

    x6 = Activation("relu")(x5c)

    ## phase branch
    # s = 512
    xc_p = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                padding="same", name='conv_p_{0}'.format(s_p))(phase_x)
    x1_p = LeakyReLU(alpha=0.2)(xc_p)
    # s = 256, nf = 128
    s_p = s_p // 2
    nf_in_p = nf_in_p * 2
    x1c_p = conv2d(nf_in_p, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s_p > 2)),
                 padding="same", name='conv_p_{0}'.format(s_p))(x1_p)
    x2_p = LeakyReLU(alpha=0.2)(x1c_p)
    # s = 128 nf = 256
    s_p = s_p // 2
    nf_in_p = nf_in_p * 2
    x2c_p = conv2d(nf_in_p, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s_p > 2)),
                 padding="same", name='conv_p_{0}'.format(s_p))(x2_p)
    x3_p = LeakyReLU(alpha=0.2)(x2c_p)
    # s = 64 nf = 512
    s_p = s_p // 2
    nf_in_p = nf_in_p * 2
    x3c_p = conv2d(nf_in_p, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s_p > 2)),
                 padding="same", name='conv_p_{0}'.format(s_p))(x3_p)
    x4_p = LeakyReLU(alpha=0.2)(x3c_p)
    # s = 32 nf = 512
    s_p = s_p // 2
    nf_in_p = nf_in_p
    x4c_p = conv2d(nf_in_p, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s_p > 2)),
                 padding="same", name='conv_p_{0}'.format(s_p))(x4_p)
    x5_p = LeakyReLU(alpha=0.2)(x4c_p)
    # s = 16 nf = 512
    s_p = s_p // 2
    nf_in_p = nf_in_p
    x5c_p = conv2d(nf_in_p, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s_p > 2)),
                 padding="same", name='conv_p_{0}'.format(s_p))(x5_p)

    x6_p = Activation('sigmoid')(x5c_p)

    #attention

    x6 = Multiply()([x6, x6_p])




    # transpose
    x6 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x6)
    x6 = Cropping2D(1)(x6)
    x4 = Concatenate(axis=channel_axis)([x6, x4c])
    x4 = Activation("relu")(x4)

    # 32, 256
    s = s * 2
    nf_in = nf_in // 2
    x4 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x4)
    x4 = Cropping2D(1)(x4)
    x3 = Concatenate(axis=channel_axis)([x4, x3c])
    x3 = Activation("relu")(x3)

    # 64, 128
    s = s * 2
    nf_in = nf_in // 2
    x3 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x3)
    x3 = Cropping2D(1)(x3)
    x2 = Concatenate(axis=channel_axis)([x3, x2c])
    x2 = Activation("relu")(x2)

    # 128, 64
    s = s * 2
    nf_in = nf_in // 2
    x2 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x2)
    x2 = Cropping2D(1)(x2)
    x1 = Concatenate(axis=channel_axis)([x2, x1c])
    x1 = Activation("relu")(x1)

    # 256, 32
    s = s * 2
    nf_in = nf_in // 2
    x1 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x1)
    x1 = Cropping2D(1)(x1)
    x = Concatenate(axis=channel_axis)([x1, xc])
    x = Activation("relu")(x)

    # 512,16
    s = s * 2
    x = Conv2DTranspose(nc_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x)
    x = Cropping2D(1)(x)

    x = Activation('tanh')(x)

    return Model(inputs=inputs, outputs=[x])

def conv_block(x,nf_in, s, use_batchnorm):
    xc = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                padding="same", name='conv_{0}'.format(s))(x)
    x1 = LeakyReLU(alpha=0.2)(xc)
    ## intensity branch
    # s = 256, nf = 128
    s = s // 2
    nf_in = nf_in * 2
    x1c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x1)
    x2 = LeakyReLU(alpha=0.2)(x1c)
    # s = 128 nf = 256
    s = s // 2
    nf_in = nf_in * 2
    x2c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x2)
    x3 = LeakyReLU(alpha=0.2)(x2c)
    # s = 64 nf = 512
    s = s // 2
    nf_in = nf_in * 2
    x3c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x3)
    x4 = LeakyReLU(alpha=0.2)(x3c)
    # s = 32 nf = 512
    s = s // 2
    nf_in = nf_in
    x4c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x4)
    x5 = LeakyReLU(alpha=0.2)(x4c)
    # s = 16 nf = 512
    s = s // 2
    nf_in = nf_in
    x5c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x5)

    x6 = LeakyReLU(alpha=0.2)(x5c)


    return x6, [xc, x1c, x2c, x3c, x4c, x5c], nf_in, s


def conv_block_deep(x,nf_in, s, use_batchnorm):
    xc = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                padding="same", name='conv_{0}'.format(s))(x)
    x1 = LeakyReLU(alpha=0.2)(xc)
    #print("xc shape",xc.shape)
    ## intensity branch
    # s = 256, nf = 128
    s = s // 2
    nf_in = nf_in * 2
    x1c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x1)
    x2 = LeakyReLU(alpha=0.2)(x1c)
    # s = 128 nf = 256
    s = s // 2
    nf_in = nf_in * 2
    x2c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x2)
    x3 = LeakyReLU(alpha=0.2)(x2c)
    # s = 64 nf = 512
    s = s // 2
    nf_in = nf_in * 2
    x3c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x3)
    x4 = LeakyReLU(alpha=0.2)(x3c)
    # s = 32 nf = 512
    s = s // 2
    nf_in = nf_in
    x4c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x4)
    x5 = LeakyReLU(alpha=0.2)(x4c)
    # s = 16 nf = 512
    s = s // 2
    nf_in = nf_in
    x5c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x5)

    x6 = LeakyReLU(alpha=0.2)(x5c)

    # s = 8 nf = 512
    s = s // 2
    nf_in = nf_in
    x6c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x6)

    x7 = LeakyReLU(alpha=0.2)(x6c)
    # s = 4 nf = 512
    s = s // 2
    nf_in = nf_in
    x7c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x7)
    #print('x7c',x7c.shape)

    x8 = LeakyReLU(alpha=0.2)(x7c)
    # s = 2 nf = 512
    s = s // 2
    nf_in = nf_in
    x8c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x8)
    #print('x8c', x8c.shape)

    x9 = LeakyReLU(alpha=0.2)(x8c)



    return x9, [xc, x1c, x2c, x3c, x4c, x5c, x6c, x7c,x8c], nf_in, s


def conv_block_phase(x,nf_in, s, use_batchnorm):
    xc = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                padding="same", name='convp_{0}'.format(s))(x)
    x1 = Activation("sigmoid")(xc)
    ## intensity branch
    # s = 256, nf = 128
    s = s // 2
    nf_in = nf_in * 2
    x1c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x1)
    x2 = Activation("sigmoid")(x1c)
    # s = 128 nf = 256
    s = s // 2
    nf_in = nf_in * 2
    x2c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x2)
    x3 = Activation("sigmoid")(x2c)
    # s = 64 nf = 512
    s = s // 2
    nf_in = nf_in * 2
    x3c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x3)
    x4 = Activation("sigmoid")(x3c)
    # s = 32 nf = 512
    s = s // 2
    nf_in = nf_in
    x4c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x4)
    x5 = Activation("sigmoid")(x4c)
    # s = 16 nf = 512
    s = s // 2
    nf_in = nf_in
    x5c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x5)

    x6 = Activation("sigmoid")(x5c)


    return x6, [xc,x1c, x2c, x3c, x4c, x5c], nf_in, s


def conv_block_phase_deep(x,nf_in, s, use_batchnorm):
    xc = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                padding="same", name='convp_{0}'.format(s))(x)
    x1 = Activation("sigmoid")(xc)
    ## intensity branch
    # s = 256, nf = 128
    s = s // 2
    nf_in = nf_in * 2
    x1c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x1)
    x2 = Activation("sigmoid")(x1c)
    # s = 128 nf = 256
    s = s // 2
    nf_in = nf_in * 2
    x2c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x2)
    x3 = Activation("sigmoid")(x2c)
    # s = 64 nf = 512
    s = s // 2
    nf_in = nf_in * 2
    x3c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x3)
    x4 = Activation("sigmoid")(x3c)
    # s = 32 nf = 512
    s = s // 2
    nf_in = nf_in
    x4c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x4)
    x5 = Activation("sigmoid")(x4c)
    # s = 16 nf = 512
    s = s // 2
    nf_in = nf_in
    x5c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x5)

    x6 = Activation("sigmoid")(x5c)
    # s = 8 nf = 512
    s = s // 2
    nf_in = nf_in
    x6c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x6)

    x7 = Activation("sigmoid")(x6c)
    # s = 4 nf = 512
    s = s // 2
    nf_in = nf_in
    x7c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x7)

    x8 = Activation("sigmoid")(x7c)
    # s = 2nf = 512
    s = s // 2
    nf_in = nf_in
    x8c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x8)

    x9 = Activation("sigmoid")(x8c)
    # s = 1nf = 512



    return x9, [xc,x1c, x2c, x3c, x4c, x5c,x6c,x7c,x8c], nf_in, s

def UNET_G_phaseattention(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    use_batchnorm = False
    s = s_p = isize
    nf_in = nf_in_p = ngf
    print('in size', s,s,nc_in)
    x = inputs = Input(shape=(s, s, nc_in))
    intens_x = crop(3,1,2)(x)
    phase_x = crop(3,2,3)(x)
    #s = 5123
    x6_p,skips_p,  _,_ = conv_block_phase(phase_x, nf_in, s, use_batchnorm)

    x10,skips, s, nf_in = conv_block_deep(intens_x, nf_in, s, use_batchnorm)
    xc, x1c, x2c, x3c, x4c, x5c, x6c, x7c,x8c,x9c = skips



    #attention

    x6 = Multiply()([x10, x10_p])



    # transpose
    x6 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x6)
    x6 = Cropping2D(1)(x6)
    x4 = Concatenate(axis=channel_axis)([x6, x4c])
    x4 = Activation("relu")(x4)

    # 32, 256
    s = s * 2
    nf_in = nf_in // 2
    x4 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x4)
    x4 = Cropping2D(1)(x4)
    x3 = Concatenate(axis=channel_axis)([x4, x3c])
    x3 = Activation("relu")(x3)

    # 64, 128
    s = s * 2
    nf_in = nf_in // 2
    x3 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x3)
    x3 = Cropping2D(1)(x3)
    x2 = Concatenate(axis=channel_axis)([x3, x2c])
    x2 = Activation("relu")(x2)

    # 128, 64
    s = s * 2
    nf_in = nf_in // 2
    x2 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x2)
    x2 = Cropping2D(1)(x2)
    x1 = Concatenate(axis=channel_axis)([x2, x1c])
    x1 = Activation("relu")(x1)

    # 256, 32
    s = s * 2
    nf_in = nf_in // 2
    x1 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x1)
    x1 = Cropping2D(1)(x1)
    x = Concatenate(axis=channel_axis)([x1, xc])
    x = Activation("relu")(x)

    # 512,16
    s = s * 2
    x = Conv2DTranspose(nc_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x)
    x = Cropping2D(1)(x)

    x = Activation('tanh')(x)

    return Model(inputs=inputs, outputs=[x])


def UNET_G_phaseatt_deep(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    use_batchnorm = False
    s = s_p = isize
    nf_in = nf_in_p = ngf
    print('in size', s,s,nc_in, nf_in)
    x = inputs = Input(shape=(s, s, nc_in))
    intens_x = crop(3,1,2)(x)
    phase_x = crop(3,2,3)(x)
    #s = 5123
    x9_p,skips_p,  _,_ = conv_block_phase_deep(phase_x, nf_in, s, use_batchnorm)
    #xc_p, x1c_p, x2c_p, x3c_p, x4c_p, x5c_p = skips_p

    x9,skips, nf_in,s = conv_block_deep(intens_x, nf_in, s, use_batchnorm)
    xc, x1c, x2c,x3c,x4c,x5c, x6c, x7c, x8c= skips



    #attention

    x9= Multiply()([x9, x9_p])



    print('x9 shape',x9.shape, nf_in)



    # transpose
    # 2, 512
    s = s * 2
    nf_in = nf_in
    x9 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x9)
    x9= Cropping2D(1)(x9)
    x7 = Concatenate(axis=channel_axis)([x9, x7c])
    x7 = Activation("relu")(x7)
    # transpose
    # 4, 512
    s = s * 2
    nf_in = nf_in
    x7 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x7)
    x7= Cropping2D(1)(x7)
    x6 = Concatenate(axis=channel_axis)([x7, x6c])
    x6 = Activation("relu")(x6)
    # transpose
    # 8, 512
    s = s * 2
    nf_in = nf_in
    x6 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x6)
    x6 = Cropping2D(1)(x6)
    x5 = Concatenate(axis=channel_axis)([x6, x5c])
    x5 = Activation("relu")(x5)

    # transpose
    # 16, 512
    s = s * 2
    nf_in = nf_in
    x5 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x5)
    x5 = Cropping2D(1)(x5)
    x4 = Concatenate(axis=channel_axis)([x5, x4c])
    x4 = Activation("relu")(x4)

    # 32, 256
    s = s * 2
    nf_in = nf_in
    x4 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x4)
    x4 = Cropping2D(1)(x4)
    x3 = Concatenate(axis=channel_axis)([x4, x3c])
    x3 = Activation("relu")(x3)

    # 64, 128
    s = s * 2
    nf_in = nf_in // 2
    x3 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x3)
    x3 = Cropping2D(1)(x3)
    x2 = Concatenate(axis=channel_axis)([x3, x2c])
    x2 = Activation("relu")(x2)

    # 128, 64
    s = s * 2
    nf_in = nf_in // 2
    x2 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x2)
    x2 = Cropping2D(1)(x2)
    x1 = Concatenate(axis=channel_axis)([x2, x1c])
    x1 = Activation("relu")(x1)

    # 256, 32
    s = s * 2
    nf_in = nf_in // 2
    x1 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x1)
    x1 = Cropping2D(1)(x1)
    x = Concatenate(axis=channel_axis)([x1, xc])
    x = Activation("relu")(x)

    # 512,16
    s = s * 2
    x = Conv2DTranspose(nc_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x)
    x = Cropping2D(1)(x)

    x = Activation('tanh')(x)

    return Model(inputs=inputs, outputs=[x])

def conv_block_ds_deep(x,nf_in, s, use_batchnorm, ds):
    x1_p, x2_p, x3_p, x4_p, x5_p, x6_p, x7_p, x8_p = ds
    xc = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                padding="same", name='conv_{0}'.format(s))(x)
    x1 = LeakyReLU(alpha=0.2)(xc)
    x1 = Multiply()([x1, x1_p])
    #print("xc shape",xc.shape)
    ## intensity branch
    # s = 256, nf = 128
    s = s // 2
    nf_in = nf_in * 2
    x1c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x1)
    x2 = LeakyReLU(alpha=0.2)(x1c)
    x2 = Multiply()([x2, x2_p])
    # s = 128 nf = 256
    s = s // 2
    nf_in = nf_in * 2
    x2c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x2)
    x3 = LeakyReLU(alpha=0.2)(x2c)
    x3 = Multiply()([x3, x3_p])
    # s = 64 nf = 512
    s = s // 2
    nf_in = nf_in * 2
    x3c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x3)
    x4 = LeakyReLU(alpha=0.2)(x3c)
    x4 = Multiply()([x4, x4_p])
    # s = 32 nf = 512
    s = s // 2
    nf_in = nf_in
    x4c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x4)
    x5 = LeakyReLU(alpha=0.2)(x4c)
    x5 = Multiply()([x5, x5_p])
    # s = 16 nf = 512
    s = s // 2
    nf_in = nf_in
    x5c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x5)

    x6 = LeakyReLU(alpha=0.2)(x5c)
    x6 = Multiply()([x6, x6_p])

    # s = 8 nf = 512
    s = s // 2
    nf_in = nf_in
    x6c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x6)

    x7 = LeakyReLU(alpha=0.2)(x6c)
    x7 = Multiply()([x7, x7_p])
    # s = 4 nf = 512
    s = s // 2
    nf_in = nf_in
    x7c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x7)
    #print('x7c',x7c.shape)

    x8 = LeakyReLU(alpha=0.2)(x7c)
    x8 = Multiply()([x8, x8_p])
    # s = 2 nf = 512
    s = s // 2
    nf_in = nf_in
    x8c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='conv_{0}'.format(s))(x8)
    #print('x8c', x8c.shape)

    x9 = LeakyReLU(alpha=0.2)(x8c)



    return x9, [xc, x1c, x2c, x3c, x4c, x5c, x6c, x7c,x8c], nf_in, s


def conv_block_phase_ds_deep(x,nf_in, s, use_batchnorm):
    xc = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                padding="same", name='convp_{0}'.format(s))(x)
    x1 = Activation("sigmoid")(xc)
    ## intensity branch
    # s = 256, nf = 128
    s = s // 2
    nf_in = nf_in * 2
    x1c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x1)
    x2 = Activation("sigmoid")(x1c)
    # s = 128 nf = 256
    s = s // 2
    nf_in = nf_in * 2
    x2c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x2)
    x3 = Activation("sigmoid")(x2c)
    # s = 64 nf = 512
    s = s // 2
    nf_in = nf_in * 2
    x3c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x3)
    x4 = Activation("sigmoid")(x3c)
    # s = 32 nf = 512
    s = s // 2
    nf_in = nf_in
    x4c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x4)
    x5 = Activation("sigmoid")(x4c)
    # s = 16 nf = 512
    s = s // 2
    nf_in = nf_in
    x5c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x5)

    x6 = Activation("sigmoid")(x5c)
    # s = 8 nf = 512
    s = s // 2
    nf_in = nf_in
    x6c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x6)

    x7 = Activation("sigmoid")(x6c)
    # s = 4 nf = 512
    s = s // 2
    nf_in = nf_in
    x7c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x7)

    x8 = Activation("sigmoid")(x7c)
    # s = 2nf = 512
    s = s // 2
    nf_in = nf_in
    x8c = conv2d(nf_in, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                 padding="same", name='convp_{0}'.format(s))(x8)

    x9 = Activation("sigmoid")(x8c)
    # s = 1nf = 512



    return x9, [x1, x2, x3, x4, x5,x6,x7,x8], nf_in, s

def UNET_G_phaseatt_ds_deep(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    use_batchnorm = False
    s = s_p = isize
    nf_in = nf_in_p = ngf
    # print('in size', s,s,nc_in, nf_in)
    x = inputs = Input(shape=(s, s, nc_in))
    intens_x = crop(3,1,2)(x)
    phase_x = crop(3,2,3)(x)
    #s = 5123
    x9_p,skips_p,  _,_ = conv_block_phase_ds_deep(phase_x, nf_in, s, use_batchnorm)
    x1_p, x2_p, x3_p, x4_p, x5_p, x6_p, x7_p,x8_p = skips_p

    x9,skips, nf_in,s = conv_block_ds_deep(intens_x, nf_in, s, use_batchnorm, [x1_p,x2_p, x3_p, x4_p, x5_p, x6_p, x7_p,x8_p])
    xc, x1c, x2c,x3c,x4c,x5c, x6c, x7c, x8c= skips



    #attention

    x9= Multiply()([x9, x9_p])



    # print('x9 shape',x9.shape, nf_in)



    # transpose
    # 2, 512
    s = s * 2
    nf_in = nf_in
    x9 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x9)
    x9= Cropping2D(1)(x9)
    x7 = Concatenate(axis=channel_axis)([x9, x7c])
    x7 = Activation("relu")(x7)
    # transpose
    # 4, 512
    s = s * 2
    nf_in = nf_in
    x7 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x7)
    x7= Cropping2D(1)(x7)
    x6 = Concatenate(axis=channel_axis)([x7, x6c])
    x6 = Activation("relu")(x6)
    # transpose
    # 8, 512
    s = s * 2
    nf_in = nf_in
    x6 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x6)
    x6 = Cropping2D(1)(x6)
    x5 = Concatenate(axis=channel_axis)([x6, x5c])
    x5 = Activation("relu")(x5)

    # transpose
    # 16, 512
    s = s * 2
    nf_in = nf_in
    x5 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x5)
    x5 = Cropping2D(1)(x5)
    x4 = Concatenate(axis=channel_axis)([x5, x4c])
    x4 = Activation("relu")(x4)

    # 32, 256
    s = s * 2
    nf_in = nf_in
    x4 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x4)
    x4 = Cropping2D(1)(x4)
    x3 = Concatenate(axis=channel_axis)([x4, x3c])
    x3 = Activation("relu")(x3)

    # 64, 128
    s = s * 2
    nf_in = nf_in // 2
    x3 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x3)
    x3 = Cropping2D(1)(x3)
    x2 = Concatenate(axis=channel_axis)([x3, x2c])
    x2 = Activation("relu")(x2)

    # 128, 64
    s = s * 2
    nf_in = nf_in // 2
    x2 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x2)
    x2 = Cropping2D(1)(x2)
    x1 = Concatenate(axis=channel_axis)([x2, x1c])
    x1 = Activation("relu")(x1)

    # 256, 32
    s = s * 2
    nf_in = nf_in // 2
    x1 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x1)
    x1 = Cropping2D(1)(x1)
    x = Concatenate(axis=channel_axis)([x1, xc])
    x = Activation("relu")(x)

    # 512,16
    s = s * 2
    x = Conv2DTranspose(nc_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x)
    x = Cropping2D(1)(x)

    x = Activation('tanh')(x)

    return Model(inputs=inputs, outputs=[x])


def UNET_G_phaseattention_deep_add(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    use_batchnorm = False
    s = s_p = isize
    nf_in = nf_in_p = ngf
    print('in size', s,s,nc_in, nf_in)
    x = inputs = Input(shape=(s, s, nc_in))
    intens_x = crop(3,1,2)(x)
    phase_x = crop(3,2,3)(x)
    #s = 5123
    x9_p,skips_p,  _,_ = conv_block_phase_deep(phase_x, nf_in, s, use_batchnorm)
    xc_p, x1c_p, x2c_p, x3c_p, x4c_p, x5c_p, x6c_p, x7c_p, x8c_p = skips_p

    x9,skips, nf_in,s = conv_block_deep(intens_x, nf_in, s, use_batchnorm)
    xc, x1c, x2c,x3c,x4c,x5c, x6c, x7c, x8c= skips



    #attention

    x9= Multiply()([x9, x9_p])
    x7c= Add()([x7c, x7c_p])
    x6c= Add()([x6c, x6c_p])
    x5c= Add()([x5c, x5c_p])
    x4c= Add()([x4c, x4c_p])
    x3c= Add()([x3c, x3c_p])
    x2c= Add()([x2c, x2c_p])
    x1c= Add()([x1c, x1c_p])
    xc = Add()([xc, xc_p])



    print('x9 shape',x9.shape, nf_in)



    # transpose
    # 2, 512
    s = s * 2
    nf_in = nf_in
    x9 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x9)
    x9= Cropping2D(1)(x9)
    x7 = Concatenate(axis=channel_axis)([x9, x7c])
    x7 = Activation("relu")(x7)
    # transpose
    # 4, 512
    s = s * 2
    nf_in = nf_in
    x7 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x7)
    x7= Cropping2D(1)(x7)
    x6 = Concatenate(axis=channel_axis)([x7, x6c])
    x6 = Activation("relu")(x6)
    # transpose
    # 8, 512
    s = s * 2
    nf_in = nf_in
    x6 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x6)
    x6 = Cropping2D(1)(x6)
    x5 = Concatenate(axis=channel_axis)([x6, x5c])
    x5 = Activation("relu")(x5)

    # transpose
    # 16, 512
    s = s * 2
    nf_in = nf_in
    x5 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x5)
    x5 = Cropping2D(1)(x5)
    x4 = Concatenate(axis=channel_axis)([x5, x4c])
    x4 = Activation("relu")(x4)

    # 32, 256
    s = s * 2
    nf_in = nf_in
    x4 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x4)
    x4 = Cropping2D(1)(x4)
    x3 = Concatenate(axis=channel_axis)([x4, x3c])
    x3 = Activation("relu")(x3)

    # 64, 128
    s = s * 2
    nf_in = nf_in // 2
    x3 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x3)
    x3 = Cropping2D(1)(x3)
    x2 = Concatenate(axis=channel_axis)([x3, x2c])
    x2 = Activation("relu")(x2)

    # 128, 64
    s = s * 2
    nf_in = nf_in // 2
    x2 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x2)
    x2 = Cropping2D(1)(x2)
    x1 = Concatenate(axis=channel_axis)([x2, x1c])
    x1 = Activation("relu")(x1)

    # 256, 32
    s = s * 2
    nf_in = nf_in // 2
    x1 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x1)
    x1 = Cropping2D(1)(x1)
    x = Concatenate(axis=channel_axis)([x1, xc])
    x = Activation("relu")(x)

    # 512,16
    s = s * 2
    x = Conv2DTranspose(nc_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x)
    x = Cropping2D(1)(x)

    x = Activation('tanh')(x)

    return Model(inputs=inputs, outputs=[x])


def UNET_G_phaseattention_deep(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    use_batchnorm = False
    s = s_p = isize
    nf_in = nf_in_p = ngf
    #print('in size', s,s,nc_in, nf_in)
    x = inputs = Input(shape=(s, s, nc_in))
    intens_x = crop(3,1,2)(x)
    phase_x = crop(3,2,3)(x)
    #s = 5123
    x9_p,skips_p,  _,_ = conv_block_phase_deep(phase_x, nf_in, s, use_batchnorm)
    xc_p, x1c_p, x2c_p, x3c_p, x4c_p, x5c_p, x6c_p, x7c_p, x8c_p = skips_p

    x9,skips, nf_in,s = conv_block_deep(intens_x, nf_in, s, use_batchnorm)
    xc, x1c, x2c,x3c,x4c,x5c, x6c, x7c, x8c= skips


    #attention

    x9= Multiply()([x9, x9_p])
    x7c= Multiply()([x7c, x7c_p])
    x6c= Multiply()([x6c, x6c_p])
    x5c= Multiply()([x5c, x5c_p])
    x4c= Multiply()([x4c, x4c_p])
    x3c= Multiply()([x3c, x3c_p])
    x2c= Multiply()([x2c, x2c_p])
    x1c= Multiply()([x1c, x1c_p])
    xc = Multiply()([xc, xc_p])



    # transpose
    # 2, 512
    s = s * 2
    nf_in = nf_in
    x9 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x9)
    x9= Cropping2D(1)(x9)
    x7 = Concatenate(axis=channel_axis)([x9, x7c])
    x7 = Activation("relu")(x7)
    # transpose
    # 4, 512
    s = s * 2
    nf_in = nf_in
    x7 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x7)
    x7= Cropping2D(1)(x7)
    x6 = Concatenate(axis=channel_axis)([x7, x6c])
    x6 = Activation("relu")(x6)
    # transpose
    # 8, 512
    s = s * 2
    nf_in = nf_in
    x6 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x6)
    x6 = Cropping2D(1)(x6)
    x5 = Concatenate(axis=channel_axis)([x6, x5c])
    x5 = Activation("relu")(x5)

    # transpose
    # 16, 512
    s = s * 2
    nf_in = nf_in
    x5 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x5)
    x5 = Cropping2D(1)(x5)
    x4 = Concatenate(axis=channel_axis)([x5, x4c])
    x4 = Activation("relu")(x4)

    # 32, 256
    s = s * 2
    nf_in = nf_in
    x4 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x4)
    x4 = Cropping2D(1)(x4)
    x3 = Concatenate(axis=channel_axis)([x4, x3c])
    x3 = Activation("relu")(x3)

    # 64, 128
    s = s * 2
    nf_in = nf_in // 2
    x3 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x3)
    x3 = Cropping2D(1)(x3)
    x2 = Concatenate(axis=channel_axis)([x3, x2c])
    x2 = Activation("relu")(x2)

    # 128, 64
    s = s * 2
    nf_in = nf_in // 2
    x2 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x2)
    x2 = Cropping2D(1)(x2)
    x1 = Concatenate(axis=channel_axis)([x2, x1c])
    x1 = Activation("relu")(x1)

    # 256, 32
    s = s * 2
    nf_in = nf_in // 2
    x1 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x1)
    x1 = Cropping2D(1)(x1)
    x = Concatenate(axis=channel_axis)([x1, xc])
    x = Activation("relu")(x)

    # 512,16
    s = s * 2
    x = Conv2DTranspose(nc_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x)
    x = Cropping2D(1)(x)

    x = Activation('tanh')(x)

    return Model(inputs=inputs, outputs=[x])



def UNET_G_deep(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    use_batchnorm = False
    s = s_p = isize
    nf_in = nf_in_p = ngf
    print('in size', s,s,nc_in, nf_in)
    x = inputs = Input(shape=(s, s, nc_in))
    intens_x = crop(3,1,2)(x)
    phase_x = crop(3,2,3)(x)
    #s = 5123

    x9,skips, nf_in,s = conv_block_deep(x, nf_in, s, use_batchnorm)
    xc, x1c, x2c,x3c,x4c,x5c, x6c, x7c, x8c= skips





    # transpose
    # 2, 512
    s = s * 2
    nf_in = nf_in
    x9 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x9)
    x9= Cropping2D(1)(x9)
    x7 = Concatenate(axis=channel_axis)([x9, x7c])
    x7 = Activation("relu")(x7)
    # transpose
    # 4, 512
    s = s * 2
    nf_in = nf_in
    x7 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x7)
    x7= Cropping2D(1)(x7)
    x6 = Concatenate(axis=channel_axis)([x7, x6c])
    x6 = Activation("relu")(x6)
    # transpose
    # 8, 512
    s = s * 2
    nf_in = nf_in
    x6 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                        kernel_initializer=conv_init,
                        name='convt.{0}'.format(s))(x6)
    x6 = Cropping2D(1)(x6)
    x5 = Concatenate(axis=channel_axis)([x6, x5c])
    x5 = Activation("relu")(x5)

    # transpose
    # 16, 512
    s = s * 2
    nf_in = nf_in
    x5 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x5)
    x5 = Cropping2D(1)(x5)
    x4 = Concatenate(axis=channel_axis)([x5, x4c])
    x4 = Activation("relu")(x4)

    # 32, 256
    s = s * 2
    nf_in = nf_in
    x4 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x4)
    x4 = Cropping2D(1)(x4)
    x3 = Concatenate(axis=channel_axis)([x4, x3c])
    x3 = Activation("relu")(x3)

    # 64, 128
    s = s * 2
    nf_in = nf_in // 2
    x3 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x3)
    x3 = Cropping2D(1)(x3)
    x2 = Concatenate(axis=channel_axis)([x3, x2c])
    x2 = Activation("relu")(x2)

    # 128, 64
    s = s * 2
    nf_in = nf_in // 2
    x2 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x2)
    x2 = Cropping2D(1)(x2)
    x1 = Concatenate(axis=channel_axis)([x2, x1c])
    x1 = Activation("relu")(x1)

    # 256, 32
    s = s * 2
    nf_in = nf_in // 2
    x1 = Conv2DTranspose(nf_in, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x1)
    x1 = Cropping2D(1)(x1)
    x = Concatenate(axis=channel_axis)([x1, xc])
    x = Activation("relu")(x)

    # 512,16
    s = s * 2
    x = Conv2DTranspose(nc_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                         kernel_initializer=conv_init,
                         name='convt.{0}'.format(s))(x)
    x = Cropping2D(1)(x)

    x = Activation('tanh')(x)

    return Model(inputs=inputs, outputs=[x])