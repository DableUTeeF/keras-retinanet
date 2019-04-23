import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras_retinanet.models import load_model
from keras import backend as K
import tensorflow as tf

K.set_learning_phase(0)
model = load_model(os.path.join('snapshots', '3', 'resnet50_csv_08-i.h5'))

saver = tf.train.Saver()
sess = K.get_session()
saver.save(sess, "./TF_Model/tf_model")

# fw = tf.summary.FileWriter('logs', sess.graph)
# fw.close()

