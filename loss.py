import keras.backend as K
import tensorflow as tf

use_lsgan = True

if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

def D_loss(netD, real, fake1, fake2, rec):
    output_real = netD([real])
    output_fake = netD([fake1])
    loss_D_real = loss_fn(output_real, K.ones_like(output_real))
    loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
    loss_G = loss_fn(output_fake, K.ones_like(output_fake))
    loss_D = loss_D_real + loss_D_fake
    loss_cyc = K.mean(K.abs(rec - real))

    fake2_Green = tf.slice(fake2, [0, 0, 0, 1], [1, 512, 512, 1])
    real_Green = tf.slice(real, [0, 0, 0, 1], [1, 512, 512, 1])
    loss_msSSIM = 1 - tf.image.ssim_multiscale(fake2_Green, real_Green, K.max(real) - K.min(real))

    return loss_D, loss_G, loss_cyc, loss_msSSIM