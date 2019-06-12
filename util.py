import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg
import tensorflow as tf
from keras.datasets import mnist


def normalize(vector, a, b, max_val=255, min_val=0):
    assert(a < b)
    print("MAX : " + str(vector.max()) + " and MIN: " + str(vector.min()))

    #result = (b - a) * ( (vector - min_val) / (max_val - min_val) ) + a
    result = (vector - 127.5) / 127.5

    print("Normalized MAX : " + str(result.max()) + " and MIN: " + str(result.min()))
    return result


def getDataTumour(path, value="mel", resize=None):
    DF = pd.read_pickle(path)
    assert(len(DF["image"]) == len(DF["id"]))
    
    X = []
    for i in range(len(DF["image"])):
        
        if DF["id"][i] == value:
            if resize is None:
                X.append(DF["image"][i])
            else:
                tmp = cv2.resize(DF["image"][i], (int(resize),int(resize)), interpolation = cv2.INTER_CUBIC)
                X.append(tmp)
                
    return np.array(X, dtype=np.float32)


def saveImages(images, epoch):    
    for i in range(len(images)):
        mpimg.imsave(base_path + "images/out-" + str(epoch) + "-" + str(i) + ".png",  ( (images[i] * 127.5) + 127.5 ).astype(np.uint8) )


def tensorboard(summaries_dir, D_loss, G_loss):
    # Summaries | VISUALIZE => tensorboard --logdir=.
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)

    merged_summary = tf.summary.merge_all()

    return merged_summary