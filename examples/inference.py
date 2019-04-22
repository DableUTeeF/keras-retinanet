# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from PIL import Image
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# use this environment flag to change which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'snapshots', 'resnet50_csv_04-i.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

# print(model.summary())

# load label to names mapping for visualization purposes
# labels_to_names = {0: 'person', 1: 'LP', 2: 'badshoes', 3: 'unsafe_hat', 4: 'Boot', 5: 'helmet',
#                    6: 'goodshoes'}
labels_to_names = {0: 'person', 1: 'helmet', 2: 'LP', 3: 'goodshoes', 4: 'badshoes'}

# load image
path = '/media/palm/data/ppa/v3/images/val/'
pad = 0
# while True:
# p = os.path.join(path, np.random.choice(os.listdir(path)))
for pth in os.listdir(path):
    p = os.path.join(path, pth)
    image = read_image_bgr(p)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=800, max_side=1333)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    ptime = time.time() - start
    print("processing time:", ptime, "nboxes:", boxes.shape)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    if ptime > 0.1:
        im = Image.fromarray(draw)
        im.save('img6/{}'.format(pth))
    else:
        im = Image.fromarray(draw)
        im.save('img6/pp{}'.format(pth))

        # plt.figure(figsize=(15, 15))
        # plt.axis('off')
        # plt.imshow(draw)
        # plt.show()
