import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def test(model, test_images, test_labels):
    loss, acc = model.evaluate(test_images, test_labels)
    ypred = model.predict(test_images)
    ypred = np.argmax(ypred, axis=1)
    return loss, test_labels, ypred

if __name__ == "__main__":

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    from train_multi_layer import create_MLP
    model_MLP = create_MLP()
    model_MLP.load_weights('./models/MLP_model.h5')

    from training_conv_net import create_LeNet
    model_conv_net = create_LeNet()
    model_conv_net.load_weights('./models/convnet_model.h5')

    loss, gt, pred = test(model_MLP, test_images, test_labels)
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    CF_MLN = confusion_matrix(gt,pred)
    plt.figure(1)
    ax=plt.subplot()
    cm_MLN = pd.DataFrame(CF_MLN, index=[i for i in range(10)],
                         columns=[i for i in range(10)])
    sns.heatmap(cm_MLN, annot=True)
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix MLN');
    ax.xaxis.set_ticklabels([f'{i}' for i in range(10)]);
    ax.yaxis.set_ticklabels([f'{i}' for i in range(10)]);

    loss, gt, pred = test(model_conv_net, test_images, test_labels)
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    CF_CNN = confusion_matrix(gt, pred)
    plt.figure(2)
    ax = plt.subplot()
    cm_CNN = pd.DataFrame(CF_CNN, index=[i for i in range(10)],
                         columns=[i for i in range(10)])
    sns.heatmap(cm_CNN, annot=True)
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix CNN');
    ax.xaxis.set_ticklabels([f'{i}' for i in range(10)]);
    ax.yaxis.set_ticklabels([f'{i}' for i in range(10)]);
    plt.show()