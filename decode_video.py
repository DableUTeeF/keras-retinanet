# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import numpy as np
import time
import pytesseract
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


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
    passed = {'head': False, 'foot': False}
    for head in heads:
        if overlaps_area(head, person) > ((head[2]-head[0])*(head[3]-head[1]))*0.5:
            passed['head'] = True
    for foot, l in foots:
        if overlaps_area(foot, person) > ((foot[2]-foot[0])*(foot[3]-foot[1]))*0.5 and l == 3:
            passed['foot'] = True
    return passed


if __name__ == '__main__':
    smin, smax = 384, 512

    keras.backend.tensorflow_backend.set_session(get_session())

    model_path = os.path.join('snapshots', '2', 'resnet50_csv_01-i.h5')

    model = models.load_model(model_path, backbone_name='resnet50')

    labels_to_names = {0: 'person', 1: 'helmet', 2: 'LP', 3: 'goodshoes', 4: 'badshoes'}
    # labels_to_names = {0: 'person', 1: 'LP', 2: 'badshoes', 3: 'unsafe_hat', 4: 'Boot', 5: 'helmet',
    #                    6: 'goodshoes'}

    # load image
    cap = cv2.VideoCapture(0)
    while True:
        stat, image = cap.read()

        # copy raw image for license plate ocr
        raw_im = image.copy()
        raw_im, _ = resize_image(raw_im, min_side=smin, max_side=smax)

        # copy to draw on
        draw = image.copy()
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=smin, max_side=smax)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        # print("processing time:", ptime, "nboxes:", boxes.shape)

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
            elif labels_to_names[label] == 'helmet' and score > 0.5:
                heads.append(box)
            # foot
            elif labels_to_names[label] in ['goodshoes', 'badshoes'] and score > 0.5:
                if labels_to_names[label] == 'goodshoes':
                    label = 3
                else:
                    label = 4
                foots.append([box, label])

        for person in people:
            if ispass(person, heads, foots) == {'head': True, 'foot': True}:
                color = label_color(6)
                caption = 'pass'
            else:
                color = label_color(0)
                caption = 'not pass'
            b = person.astype(int)
            draw_box(draw, b, color=color)

            draw_caption(draw, b, caption)

        for instance in (*heads, *foots):
            if len(instance) == 4:
                color = label_color(1)
                caption = 'helmet'
                b = instance.astype(int)
            else:
                color = label_color(int(instance[1]))
                caption = ['goodshoes', 'badshoes'][int(instance[1])-3]
                b = instance[0].astype(int)
            draw_box(draw, b, color=color)

            draw_caption(draw, b, caption)

        for plate in plates:
            x1, y1, x2, y2 = plate.astype('uint8')
            caption = 'LP'
            if x1 < x2 and y1 < y2:
                im = raw_im[y1:y2, x1:x2, :]
                caption = pytesseract.image_to_string(im)
            color = label_color(3)
            draw_box(draw, b, color=color)
            draw_caption(draw, b, caption)

        ptime = time.time() - start
        cv2.putText(draw, '{:.3}ms'.format(ptime*1000), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 255))
        cv2.imshow('frame', draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
