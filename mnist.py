import numpy as np

from Cortex import *

from functools import reduce
from tensorflow.examples.tutorials.mnist import input_data


io_path = 'io/'
ex_path = io_path + 'mnist/'
data_path = ex_path + 'data'
logs_path = ex_path + 'logs'
model_filename = ex_path + 'model'


def get_acc(lab, out):
    correct = [int(np.argmax(lab[n]) == np.argmax(out[n])) for n in range(len(lab))]
    return reduce(int.__add__, correct) / len(correct)


#Data

print("Preparing data...")
mnist_data = input_data.read_data_sets(data_path, one_hot = True)
print("Data info --- ")
print("Training data size: ", mnist_data.train.num_examples)
print("Testing data size: ", mnist_data.test.num_examples)


#HyperParas

dropout_rate = 0.5

learning_rate = 0.001


#Network

nn = Network()

nn.add_placeholder('img_src', [None, 28 * 28 * 1])
nn.add_placeholder('lab', [None, 10])

nn.add_layer_make_img('MkImg', 'img_src', 'img', [28, 28, 1])
nn.add_layer_conv_2d('Conv1', 'img', 'c1', [3, 3], 8, act_func = ActivationFunc.leaky_relu, batch_norm = True)
nn.add_layer_pool_2d('Pool1', 'c1', 'p1', [2, 2])
nn.add_layer_conv_2d('Conv2', 'p1', 'c2', [3, 3], 16, act_func = ActivationFunc.leaky_relu, batch_norm = True)
nn.add_layer_pool_2d('Pool2', 'c2', 'p2', [2, 2])
nn.add_layer_flat_img('FlatImg', 'p2', 'p2_flat')

nn.add_layer_full_conn('FC1', 'p2_flat', 'h', 400, act_func = ActivationFunc.leaky_relu,
                       dropout_rate_name = 'dropout_rate', batch_norm = True)
nn.add_layer_full_conn('FC2', 'h', 'h1', 100, act_func = ActivationFunc.leaky_relu,
                       dropout_rate_name = 'dropout_rate', batch_norm = True)
nn.add_layer_full_conn('FC3', 'h1', 'out', 10, act_func = ActivationFunc.softmax, batch_norm = True)

nn.add_reduce_average_cross_entropy('ace', 'lab', 'out', log_scalar = True)

nn.set_trainer(Optimizer.Adam, 'ace')


#Session

nn.logs_path(logs_path, clear = True, make_tensorbord_runner = True)
nn.print_graph()

essential_summary = nn.get_pull_essential_summary()

nn.session_new()

if nn.session_import_variables_from_file(model_filename) == {}:
    print('No saved model is found. Start with initial values.')
else:
    print('A saved model is found. Start with it.')


for step in range(10000):

    train_img_src, train_lab = mnist_data.train.next_batch(100)

    if step % 500 == 0:
        train_out_value, summary = nn.session_pull(['out', essential_summary],
                                                  feed_dict = {'img_src':train_img_src, 'lab':train_lab},
                                                  dropout_rate_dict = {'dropout_rate': 0})
        train_acc = get_acc(train_lab, train_out_value)
        nn.print_summary(summary, step, 'train')

        test_img_src, test_lab = mnist_data.test.next_batch(100)
        test_out_value, summary = nn.session_pull(['out', essential_summary],
                                                 feed_dict = {'img_src':test_img_src, 'lab':test_lab},
                                                 dropout_rate_dict = {'dropout_rate':0})
        test_acc = get_acc(test_lab, test_out_value)
        nn.print_summary(summary, step, 'test')

        print('Step %d: train_acc = %.1f%%; test_acc = %.1f%%.' % (step, train_acc * 100, test_acc * 100))

    nn.session_train(learning_rate = learning_rate,
                        feed_dict = {'img_src':train_img_src, 'lab':train_lab},
                        dropout_rate_dict = {'dropout_rate': dropout_rate})

if nn.session_export_variables_to_file(model_filename) is None:
    print('Fail to save the model.')
else:
    print('Model saved.')

nn.session_close()



