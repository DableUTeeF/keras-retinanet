# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from PIL import Image
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import time
import pytesseract
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.config import parse_anchor_parameters
from keras_retinanet.utils.model import freeze as freeze_model


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def overlaps_area(box1, box2):
    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
    return intersect_w * intersect_h


def ispass(person, heads, foots):
    passed = {'head': 0, 'foot': 0}
    x1, y1, x2, y2 = person
    for head, l, s in heads:
        if overlaps_area(head, person) > ((head[2] - head[0]) * (head[3] - head[1])) * 0.5 and \
                (y1+((y2-y1)*0.3)) > (head[3] - (head[3]-head[1])/2) > y1:  # todo: these line still bugging
            if l == 1:
                passed['head'] += s
            elif l == 2:
                passed['head'] -= s
    for foot, l, s in foots:
        if overlaps_area(foot, person) > ((foot[2] - foot[0]) * (foot[3] - foot[1])) * 0.5 and \
                person[3] > (foot[3] - (foot[3]-foot[1])/2) > (person[3]-((person[3]-person[1])*0.3)):
            if l == 3:
                passed['foot'] += s
            else:
                passed['foot'] -= s
    return passed


def create_models(backbone_retinanet, num_classes, freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model

    model = backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier)
    training_model = model
    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model


if __name__ == '__main__':
    smin, smax = 618, 800

    keras.backend.tensorflow_backend.set_session(get_session())

    model_path = os.path.join('snapshots', '7', 'resnet18_csv_19.h5')
    backbone = models.backbone('resnet18')

    labels_to_names = {0: 'goodhelmet', 1: 'LP', 2: 'goodshoes', 3: 'badshoes', 4: 'badhelmet', 5: 'person'}
    # labels_to_names = {0: 'person', 1: 'helmet', 2: 'LP', 3: 'goodshoes', 4: 'badshoes'}
    # labels_to_names = {0: 'person', 1: 'LP', 2: 'badshoes', 3: 'unsafe_hat', 4: 'Boot', 5: 'helmet',
    #                    6: 'goodshoes'}
    main_model, training_model, prediction_model = create_models(backbone.retinanet, len(labels_to_names))
    main_model.load_weights(model_path)
    model = prediction_model
    # load image
    path = '/media/palm/data/ppa/v6/images/val/'
    pad = 0
    # while True:
    # p = os.path.join(path, np.random.choice(os.listdir(path)))
    for pth in os.listdir(path):
        p = os.path.join(path, pth)
        image = read_image_bgr(p)

        # copy raw image for license plate ocr
        raw_im = image.copy()

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        # image = preprocess_image(image)
        image, scale = resize_image(image, min_side=smin, max_side=smax)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        ptime = time.time() - start
        print("processing time:", ptime*1000, 'ms')

        # correct for image scale
        boxes /= scale

        people = []
        heads = []
        foots = []
        plates = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # box       : A list of 4 elements (x1, y1, x2, y2).
            # people
            if label == -1:
                continue
            if labels_to_names[label] == 'person' and score > 0.5:
                people.append(box)
            # license plate
            elif labels_to_names[label] == 'LP' and score > 0.5:
                plates.append(box)
            # head
            elif labels_to_names[label] in ['goodhelmet', 'badhelmet'] and score > 0.5:
                if labels_to_names[label] == 'goodhelmet':
                    label = 1
                else:
                    label = 2
                heads.append([box, label, score])
            # foot
            elif labels_to_names[label] in ['goodshoes', 'badshoes'] and score > 0.5:
                if labels_to_names[label] == 'goodshoes':
                    label = 3
                else:
                    label = 4
                foots.append([box, label, score])

        for person in people:
            # if ispass(person, heads, foots) == {'head': True, 'foot': True}:
            if ispass(person, heads, foots)['head'] > 0 and ispass(person, heads, foots)['foot'] > 0:
                color = (0, 255, 0)
                caption = 'pass'
            else:
                color = (255, 0, 0)
                caption = 'not pass'
            b = person.astype(int)
            draw_box(draw, b, color=color)

            draw_caption(draw, b, caption)

        for instance in (*heads, *foots):
            color = [(0, 255, 128), (255, 0, 128), (128, 255, 0), (255, 128, 0), ][int(instance[1]) - 1]
            caption = ['goodhelmet', 'badhelmet', 'goodshoes', 'badshoes', ][int(instance[1]) - 1]
            b = instance[0].astype(int)
            draw_box(draw, b, color=color)

            draw_caption(draw, b, caption)
        a = 0
        for plate in plates:
            x1, y1, x2, y2 = plate.astype(int)
            caption = 'LP'
            if x1 < x2 and y1 < y2:
                im = raw_im[y1:y2, x1:x2]
                im, _ = resize_image(im, min_side=96, max_side=256)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                i = Image.fromarray(im)
                a += 1
                i.save('examples/img14/' + pth + str(a) + '.jpg')
                caption = pytesseract.image_to_string(im)
            color = (255, 128, 0)
            draw_box(draw, plate.astype(int), color=color)
            draw_caption(draw, plate.astype(int), caption)

        img = Image.fromarray(draw)
        img.save('examples/img14/' + pth)
