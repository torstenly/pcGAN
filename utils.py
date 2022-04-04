import keras.backend as K
from PIL import Image
import numpy as np
from random import randint, shuffle
from IPython.display import display
import glob


loadSize = 512
imageSize = 512
channel_first = False

def rescale(img):
    min_v = np.min(img)
    max_v = np.max(img)
    img = (img - min_v) / (max_v - min_v) * 255
    #img = img.astype(np.int8)
    return img


def minibatch(data, batchsize):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batchsize
        if i + size > length:
            shuffle(data)
            i = 0
            epoch += 1
        rtn = [read_image(data[j]) for j in range(i, i + size)]
        i += size
        tmpsize = yield epoch, np.float32(rtn)

        
def read_image(fn):
    im = Image.open(fn).convert('RGB')
    im = im.resize((loadSize, loadSize), Image.BILINEAR)
    arr = np.array(im) / 255 * 2 - 1
    w1, w2 = (loadSize - imageSize) // 2, (loadSize + imageSize) // 2
    h1, h2 = w1, w2
    img = arr[h1:h2, w1:w2, :]
    if channel_first:
        img = np.moveaxis(img, 2, 0)
    return img


def showG2(A):
    def G(fn_generate, X):
        r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
        return r.swapaxes(0,1)[:,:,0]
    rA = G(cycleA_generate, A)
    arr = np.concatenate([A,rA[0]])
    showX(arr, 2)

def showG(A,B):
    assert A.shape==B.shape
    def G(fn_generate, X):
        r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
        return r.swapaxes(0,1)[:,:,0]
    rA = G(cycleA_generate, A)
    rB = G(cycleB_generate, B)
    arr = np.concatenate([A,B,rA[0],rB[0],rA[1],rB[1]])
    showX(arr, 3)


def showX(X, rows=1):
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1,3,imageSize,imageSize), 1, 3)
    else:
        int_X = int_X.reshape(-1,imageSize,imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    display(Image.fromarray(int_X))


def minibatchAB(dataA, dataB, batchsize):
    batchA = minibatch(dataA, batchsize)
    batchB = minibatch(dataB, batchsize)
    tmpsize = None
    while True:
        ep1, A = batchA.send(tmpsize)
        ep2, B = batchB.send(tmpsize)
        tmpsize = yield max(ep1, ep2), A, B


def cycle_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    indentity_output = netG2([real_input]) # G(y) or F(x)
    fn_generate = K.function([real_input], [fake_output, rec_input])
    return real_input, fake_output, rec_input, indentity_output, fn_generate


def load_data(file_pattern):
    return glob.glob(file_pattern)


def G(fn_generate, X):
    r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
    return r.swapaxes(0,1)[:,:,0]