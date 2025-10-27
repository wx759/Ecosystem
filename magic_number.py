import numpy
import random

from Cortex import *
from functools import reduce
# from functools import reducesession_path


io_path = 'io/'
ex_path = io_path + 'magic_number/'
logs_path = ex_path + 'logs'
model_filename = ex_path + 'model'

def make_lab(X, count_len, lag_len, n):
    def __add_l(s, l):
        if type(s)==list: s = reduce(int.__add__, s)
        return s + reduce(int.__add__, l)
    t0 = count_len + lag_len - 1
    num = [[reduce(__add_l, s[t-t0 : t-lag_len+1]) % n for t in range(t0, len(s))] for s in X]
    return [[[1 if i==o else 0 for i in range(n)] for o in s] for s in num]

def get_random_batch(binary, batch_size, seq_len, count_len, lag_len, n):
    sel = [random.randint(0, len(binary) - seq_len) for i in range(batch_size)]
    X = [[binary[t : t+1] for t in range(i, i + seq_len)] for i in sel]
    lab = make_lab(X, count_len, lag_len, n)
    return X, lab

def long_seq_test(seq_len):
    long_seq = [random.randint(0, 1) for i in range(10000)]
    long_seq_test_X, long_seq_test_lab = get_random_batch(long_seq, 1, seq_len, count_len, lag_len, n)
    long_seq_test_lab_num = [numpy.argmax(o) for o in long_seq_test_lab[0]]
    long_seq_test_out_tail = nn.session_pull('out_tail', sequence_length = [seq_len],
                                             feed_dict = {'X':long_seq_test_X, 'lab':long_seq_test_lab})
    long_seq_test_out_num = [numpy.argmax(o) for o in long_seq_test_out_tail[0]]
    long_seq_test_x = [x[0] for x in long_seq_test_X[0]]
    hit = [1 if long_seq_test_lab_num[i] == long_seq_test_out_num[i] else 0 for i in range(len(long_seq_test_lab_num))]
    acc = reduce(int.__add__, hit) / len(hit)
    hit_str = ''
    for i in hit: hit_str += 'O' if i else 'X'
    print('----------------------------Long_Seq_Test-------------------------------')
    print('Hit:')
    print(hit_str)
    print('Acc = %.2f%%' % (acc * 100))
    print('----------------------------------------------------------------------------')


#Data

train_binary = [random.randint(0, 1) for _ in range(100000)]
test_binary = [random.randint(0, 1) for _ in range(100000)]


#HyperParas

learning_rate = 0.01
batch_size = 200
train_seq_len = 250
test_seq_len = 250
test_seq_len_long = 2500
n = 5
count_len = 20
lag_len = 5


#Network

nn = Network()

nn.add_placeholder('X', [None, None, 1])
nn.add_placeholder('lab', [None, None, n])

nn.add_layer_lstm('LSTM1', 'X', 'H1', n * 4)
nn.add_layer_batch_norm('BN1', 'H1', 'H1_norm')
nn.add_layer_full_conn('FC1', 'H1_norm', 'H2', n * 2, act_func = ActivationFunc.leaky_relu, batch_norm = True)
nn.add_layer_full_conn('FC2', 'H2', 'out', n, act_func = ActivationFunc.softmax, batch_norm = True)

nn.add_layer_sequence_tail('Tail', 'out', 'out_tail', 'lab')
nn.add_reduce_average_cross_entropy('ace', 'lab', 'out_tail', True)
nn.add_reduce_classifier_accuracy('acc', 'lab', 'out_tail', True)
nn.set_trainer(Optimizer.Adam, 'ace')

nn.logs_path(logs_path, clear = True, make_tensorbord_runner = True)
nn.print_graph()

essential_summary = nn.get_pull_essential_summary()

#Session

nn.session_new()

if nn.session_import_variables_from_file(model_filename) == {}:
    print('No saved model is found. Start with initial values.')
else:
    print('A saved model is found. Start with it.')

for step in range(2000):

    train_X, train_lab = get_random_batch(train_binary, batch_size, train_seq_len, count_len, lag_len, n)
    test_X, test_lab = get_random_batch(test_binary, batch_size, test_seq_len, count_len, lag_len, n)
    train_sequence_length = [train_seq_len] * batch_size
    test_sequence_length = [test_seq_len] * batch_size

    if step % 25 == 0:
        train_acc, summary = nn.session_pull(['acc', essential_summary], sequence_length = train_sequence_length,
                                            feed_dict = {'X':train_X, 'lab':train_lab})
        nn.print_summary(summary, step, 'train')
        test_acc, summary = nn.session_pull(['acc', essential_summary], sequence_length = test_sequence_length,
                                           feed_dict = {'X':test_X, 'lab':test_lab})
        nn.print_summary(summary, step, 'test')

        print('Step %d: train_acc = %.2f%%; test_acc = %.2f%%' % (step, train_acc * 100, test_acc * 100))

        if (train_acc + test_acc) / 2 > 0.99: break

    if step % 500 == 0:
        long_seq_test(test_seq_len_long)

    if (step + 1) % 500 == 0:
        if nn.session_export_variables_to_file(model_filename) is None:
            print('Fail to save the model.')
        else:
            print('Model saved.')

    nn.session_train(learning_rate = learning_rate, feed_dict = {'X':train_X, 'lab':train_lab},
                        sequence_length = train_sequence_length)

long_seq_test(test_seq_len_long)

if nn.session_export_variables_to_file(model_filename) is None:
    print('Fail to save the model.')
else:
    print('Model saved.')

nn.session_close()





