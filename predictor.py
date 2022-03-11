import sys

import numpy as np
import time
from load_model.network import *
from load_model.layer import *
sys.path.append("../")
from group_fairness_metric import statistical_parity_difference, equality_of_oppo
from group_fairness_metric import disparte_impact



def dnn(input_shape=(None, 13), nb_classes=2):
    """
    The implementation of a DNN model
    :param input_shape: the shape of dataset
    :param nb_classes: the number of classes
    :return: a DNN model
    """
    activation = ReLU
    layers = [Linear(64),
              activation(),
              Linear(32),
              activation(),
              Linear(16),
              activation(),
              Linear(8),
              activation(),
              Linear(4),
              activation(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def gradient_graph(x, preds, y=None):
    """
    Construct the TF graph of gradient
    :param x: the input placeholder
    :param preds: the model's symbolic output
    :return: the gradient graph
    """
    if y == None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = model_loss(y, preds, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    return grad

def model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    if op.type == "Softmax":
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out

def model_argmax(sess, x, predictions, samples, feed=None):
    """
    Helper function that computes the current class prediction
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param samples: numpy array with input samples (dims must match x)
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance
    :return: the argmax output of predictions, i.e. the current predicted class
    """
    feed_dict = {x: samples}
    if feed is not None:
        feed_dict.update(feed)
    probabilities = sess.run(predictions, feed_dict)

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)




def get_predicted_tuple(dataset=None, model_path=None, id_list_cnt=0):
    id_list = ['01', '02', '03', '04', '05']
    id_list_cnt = id_list_cnt
    if dataset is None:
        dataset = 'adult'
        input_shape = (None, 13)
        model_path = './model/' + 'test.model'
        test_instances_array = np.load('data/test_instances_set' + id_list[id_list_cnt] + '.npy',
                                       allow_pickle=True)
    else:
        dataset = 'bank'
        input_shape = (None, 20)
        model_path = './bank-additional/model/bank-additional/999/' + 'test.model'
        test_instances_array = np.load('test_instance_set/bank_test_instances_set' + id_list[id_list_cnt] + '.npy',
                                       allow_pickle=True)
    nb_classes = 2
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    grad_0 = gradient_graph(x, preds)
    X_tuple = []
    y_tuple = []
    y_pre_tuple = []  # 5 results' list
    test_accu = []
    y_predict_20 = []  # 20组 y_predict shape[0]应该20
    for i in range(20):
        print('starting round %d' % i)
        accuracy = 0.
        cor_cnt = 0
        wro_cnt = 0
        sample_id = 1
        y_pre = []
        y_true = []
        X_sample = []
        for idx in test_instances_array[i]:
            if dataset == 'adult':
                sample_tmp = np.load('./data/data-x.npy')[idx]
                ground_truth_tmp_array = np.load('./data/data-y.npy')[idx]
            else:
                sample_tmp = np.load('./bank-additional/data/data-bank-additional-X1.npy')[idx]
                ground_truth_tmp_array = np.load('./bank-additional/data/data-bank-additional-Y1.npy')[idx]
            label_tmp = model_argmax(sess, x, preds, np.array([sample_tmp]))
            X_sample.append(sample_tmp)  # 保存当前的instance
            y_pre.append(label_tmp)  # 保存当前推理获得的y值 0 :<50k 1: >50k
            if ground_truth_tmp_array[0] > 0:
                ground_truth_tmp = 0
            else:
                ground_truth_tmp = 1
            y_true.append(ground_truth_tmp)

            if label_tmp == ground_truth_tmp:
                cor_cnt += 1
            else:
                wro_cnt += 1
            sample_id += 1
        X_arr = np.array(X_sample, dtype=np.float32)
        y_arr = np.array(y_pre, dtype=np.float32)
        y_true_arr = np.array(y_true, dtype=np.float32)
        X_tuple.append(X_arr)
        y_tuple.append(y_arr)
        y_pre_tuple.append(y_true_arr)
        y_predict_20.append(y_pre)
        accuracy = cor_cnt / (cor_cnt + wro_cnt)
        test_accu.append(accuracy)
        print("test id: %d total accuracy is %f" % (i + 1, accuracy))
    np.save('adult-res-raw/adult-x' + id_list[id_list_cnt] + '.npy', np.array(X_tuple, dtype=np.float32))
    np.save('adult-res-raw/adult-y-true' + id_list[id_list_cnt] + '.npy', np.array(y_tuple, dtype=np.float32))
    np.save('adult-res-raw/adult-y-pre' + id_list[id_list_cnt] + '.npy', np.array(y_pre_tuple, dtype=np.float32))
    return X_tuple, y_tuple, y_pre_tuple, test_accu


def get_5_results():
    tuple_res = []
    for i in range(5):
        tuple_res.append(get_predicted_tuple(id_list_cnt=i))
    return tuple_res


if __name__ == '__main__':
    get_5_results()
