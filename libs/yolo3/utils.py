"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image, ImageEnhance
import cv2
import numpy as np

import keras.backend as K
from keras.layers import Input, Lambda, add
from keras.models import Model
from  .. import config
import re

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    image = np.array(image, dtype=np.uint8)
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    dx, dy = ((w-nw)//2, (h-nh)//2)
    image = cv2.resize(image, (nw,nh), cv2.INTER_LINEAR)
    
    new_image = np.zeros((h,w,config.time_step), dtype=np.uint8)
    target_new = new_image[max(0, dy):,max(0,dx):]
    target = image[max(0, -dy):,max(0,-dx):]
    target_new[:target.shape[0],:target.shape[1]] = target[:target_new.shape[0],:target_new.shape[1]]

    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=False, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    if config.use_opti:
        im_id = int((line[0])[:-4].replace('_', '/').split('/')[-1])
        im = cv2.imread("/home/zya/data/rat_b/"+line[0], cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        image = np.empty((h, w, config.time_step))
        # image[..., config.time_step//2] = im
        for i in range(-config.time_step//2, config.time_step-config.time_step//2):
            im_tmp = cv2.imread("/home/zya/data/rat_b/"+line[0].replace(f"{im_id}.", f"{im_id+i*3}."), cv2.IMREAD_GRAYSCALE)
            image[..., i+config.time_step//2] = im_tmp if im_tmp is not None else im
    else:
        image = cv2.imread("/home/zya/data/rat_b/"+line[0])
    if config.data_arg:
        image = Image.fromarray(image)
        if rand()<0.25: image = ImageEnhance.Brightness(image).enhance(np.exp(np.random.normal(0,.5)))
        if rand()<0.25: image = ImageEnhance.Contrast(image).enhance(np.exp(np.random.normal(0,.5)))
        if rand()<0.25: image = ImageEnhance.Sharpness(image).enhance(np.exp(np.random.normal(0,.5)))
        image = np.asarray(image, dtype=np.uint8)
    flipt = 0 #rand()<.5
    if flipt:
        image = image.transpose(1,0,2)

    ih, iw, _ = image.shape
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    if box.shape[1] == 5:
        box = np.concatenate([
            box[:,:4],
            np.zeros((box.shape[0],2)),
            np.expand_dims(box[:, 4], -1)
        ], axis=-1)

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        dx, dy = ((w-nw)//2, (h-nh)//2)
        image = cv2.resize(image, (nw,nh), cv2.INTER_LINEAR)
        
        new_image = np.zeros((h,w,config.time_step), dtype=np.uint8)
        target_new = new_image[max(0, dy):,max(0,dx):]
        target = image[max(0, -dy):,max(0,-dx):]
        target_new[:target.shape[0],:target.shape[1]] = target[:target_new.shape[0],:target_new.shape[1]]
        image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,7))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.75, 1.25)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)

    image = cv2.resize(image, (nw,nh), cv2.INTER_LINEAR)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))

    new_image = np.zeros((h,w,config.time_step), dtype=np.uint8)
    target_new = new_image[max(0, dy):,max(0,dx):]
    target = image[max(0, -dy):,max(0,-dx):]
    target_new[:target.shape[0],:target.shape[1]] = target[:target_new.shape[0],:target_new.shape[1]]

    image = new_image

    # flip image or not
    flip = 0 #rand()<.5
    if flip:
        image = image[:,::-1]
    flipv = 0 #rand()<.5
    if flipv:
        image = image[::-1, :]

    rand_c = 0 #rand() < .25 if not config.use_opti else 0
    if rand_c: image = image[:,:,np.random.permutation(3)]
    # inv_c = rand()<.1
    # if inv_c: image = 255-image

    # distort image
    # hue = rand(-hue, hue)
    # sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    # val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    # x = rgb_to_hsv(np.array(image)/255.)
    # x[..., 0] += hue
    # x[..., 0][x[..., 0]>1] -= 1
    # x[..., 0][x[..., 0]<0] += 1
    # x[..., 1] *= sat
    # x[..., 2] *= val
    # x[x>1] = 1
    # x[x<0] = 0
    # image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    image = np.array(image, dtype='float32')/255.
    if config.noise and rand()<0.1: image = np.power(image, max(0.1, np.random.normal(1,0.25)))
    if config.noise and rand()<0.1: image += np.random.normal(0, 1./16, size=image.shape)

    # correct boxes
    box_data = np.zeros((max_boxes,7))
    if len(box)>0:
        np.random.shuffle(box)
        if flipt:
            box[:, [0,1,2,3,4,5]] = box[:, [1,0,3,2,5,4]]
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip:
            box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 5] = -box[:, 5]
        if flipv:
            box[:, [1,3]] = h - box[:, [3,1]]
            box[:, 4] = -box[:, 4]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    box_data[:,4:6] /= 1000.
    return image, box_data


# From: https://github.com/anvien/Multi-YOLOv3/blob/master/train_YOLOv3_MultiGPU.py
'''modify the devices('/cpu:0') concatenate to add for yolo loss tensor(1,)'''
def multi_gpu_model(model, gpus=None):
    """Replicates a model on different GPUs.
    Specifically, this function implements single-machine
    multi-GPU data parallelism. It works in the following way:
    - Divide the model's input(s) into multiple sub-batches.
    - Apply a model copy on each sub-batch. Every model copy
        is executed on a dedicated GPU.
    - Concatenate the results (on CPU) into one big batch.
    E.g. if your `batch_size` is 64 and you use `gpus=2`,
    then we will divide the input into 2 sub-batches of 32 samples,
    process each sub-batch on one GPU, then return the full
    batch of 64 processed samples.
    This induces quasi-linear speedup on up to 8 GPUs.
    This function is only available with the TensorFlow backend
    for the time being.
    # Arguments
        model: A Keras model instance. To avoid OOM errors,
            this model could have been built on CPU, for instance
            (see usage example below).
        gpus: Integer >= 2 or list of integers, number of GPUs or
            list of GPU IDs on which to create model replicas.
    # Returns
        A Keras `Model` instance which can be used just like the initial
        `model` argument, but which distributes its workload on multiple GPUs.
    # Example
    ```python
        import tensorflow as tf
        from keras.applications import Xception
        from keras.utils import multi_gpu_model
        import numpy as np
        num_samples = 1000
        height = 224
        width = 224
        num_classes = 1000
        # Instantiate the base model (or "template" model).
        # We recommend doing this with under a CPU device scope,
        # so that the model's weights are hosted on CPU memory.
        # Otherwise they may end up hosted on a GPU, which would
        # complicate weight sharing.
        with tf.device('/cpu:0'):
            model = Xception(weights=None,
                             input_shape=(height, width, 3),
                             classes=num_classes)
        # Replicates the model on 8 GPUs.
        # This assumes that your machine has 8 available GPUs.
        parallel_model = multi_gpu_model(model, gpus=8)
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop')
        # Generate dummy data.
        x = np.random.random((num_samples, height, width, 3))
        y = np.random.random((num_samples, num_classes))
        # This `fit` call will be distributed on 8 GPUs.
        # Since the batch size is 256, each GPU will process 32 samples.
        parallel_model.fit(x, y, epochs=20, batch_size=256)
        # Save model via the template model (which shares the same weights):
        model.save('my_model.h5')
    ```
    # On model saving
    To save the multi-gpu model, use `.save(fname)` or `.save_weights(fname)`
    with the template model (the argument you passed to `multi_gpu_model`),
    rather than the model returned by `multi_gpu_model`.
    """

    def _get_available_devices():
        return [x.name for x in K.get_session().list_devices()]

    def _normalize_device_name(name):
        name = '/' + ':'.join(name.lower().replace('/', '').split(':')[-2:])
        return name
    
    if K.backend() != 'tensorflow':
        raise ValueError('`multi_gpu_model` is only available '
                         'with the TensorFlow backend.')

    available_devices = _get_available_devices()
    available_devices = [_normalize_device_name(name) for name in available_devices]
    if not gpus:
        # Using all visible GPUs when not specifying `gpus`
        # e.g. CUDA_VISIBLE_DEVICES=0,2 python3 keras_mgpu.py
        gpus = len([x for x in available_devices if 'gpu' in x])

    if isinstance(gpus, (list, tuple)):
        if len(gpus) <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `len(gpus) >= 2`. '
                             'Received: `gpus=%s`' % gpus)
        num_gpus = len(gpus)
        target_gpu_ids = gpus
    else:
        if gpus <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `gpus >= 2`. '
                             'Received: `gpus=%d`' % gpus)
        num_gpus = gpus
        target_gpu_ids = range(num_gpus)

    import tensorflow as tf

    target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in target_gpu_ids]
    for device in target_devices:
        if device not in available_devices:
            raise ValueError(
                'To call `multi_gpu_model` with `gpus=%d`, '
                'we expect the following devices to be available: %s. '
                'However this machine only has: %s. '
                'Try reducing `gpus`.' % (gpus,
                                          target_devices,
                                          available_devices))

    def get_slice(data, i, parts):
        shape = tf.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == num_gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        return tf.slice(data, start, size)

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for i, gpu_id in enumerate(target_gpu_ids):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('replica_%d' % gpu_id):
                inputs = []
                # Retrieve a slice of the input.
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_i = Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'i': i,
                                                'parts': num_gpus})(x)
                    inputs.append(slice_i)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/cpu:0'):
        merged = []
        for name, outputs in zip(model.output_names, all_outputs):
            merged.append(add(outputs, name=name))
        return Model(model.inputs, merged)
