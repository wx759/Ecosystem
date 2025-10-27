# # import tensorflow as tf
# import numpy as np
# # from baselines.common.tf_util import get_session
# class RunningMeanStd(object):
#     # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#     def __init__(self, epsilon=1e-4, shape=()):
#         self.mean = np.zeros(shape, 'float64')
#         self.var = np.ones(shape, 'float64')
#         self.count = epsilon
#
#     def update(self, x):
#         batch_mean = np.mean(x, axis=0)
#         batch_var = np.var(x, axis=0)
#         batch_count = x.shape[0]
#         self.update_from_moments(batch_mean, batch_var, batch_count)
#
#     def update_from_moments(self, batch_mean, batch_var, batch_count):
#         self.mean, self.var, self.count = update_mean_var_count_from_moments(
#             self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
#
# def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
#     delta = batch_mean - mean
#     tot_count = count + batch_count
#
#     new_mean = mean + delta * batch_count / tot_count
#     m_a = var * count
#     m_b = batch_var * batch_count
#     M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
#     new_var = M2 / tot_count
#     new_count = tot_count
#
#     return new_mean, new_var, new_count
# '''
# class TfRunningMeanStd(object):
#     # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#
#     TensorFlow variables-based implmentation of computing running mean and std
#     Benefit of this implementation is that it can be saved / loaded together with the tensorflow model
#
#     def __init__(self, epsilon=1e-4, shape=(), scope='',sess=tf.Session()):
#
#         self._new_mean = tf.placeholder(shape=shape, dtype=tf.float64)
#         self._new_var = tf.placeholder(shape=shape, dtype=tf.float64)
#         self._new_count = tf.placeholder(shape=(), dtype=tf.float64)
#
#
#         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#             self._mean  = tf.get_variable('mean',  initializer=np.zeros(shape, 'float64'),      dtype=tf.float64)
#             self._var   = tf.get_variable('std',   initializer=np.ones(shape, 'float64'),       dtype=tf.float64)
#             self._count = tf.get_variable('count', initializer=np.full((), epsilon, 'float64'), dtype=tf.float64)
#
#         self.update_ops = tf.group([
#             self._var.assign(self._new_var),
#             self._mean.assign(self._new_mean),
#             self._count.assign(self._new_count)
#         ])
#
#         sess.run(tf.variables_initializer([self._mean, self._var, self._count]))
#         self.sess = sess
#         self._set_mean_var_count()
#
#     def _set_mean_var_count(self):
#         self.mean, self.var, self.count = self.sess.run([self._mean, self._var, self._count])
#
#     def update(self, x):
#         batch_mean = np.mean(x, axis=0)
#         batch_var = np.var(x, axis=0)
#         batch_count = x.shape[0]
#
#         new_mean, new_var, new_count = update_mean_var_count_from_moments(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
#
#         self.sess.run(self.update_ops, feed_dict={
#             self._new_mean: new_mean,
#             self._new_var: new_var,
#             self._new_count: new_count
#         })
#
#         self._set_mean_var_count()
# '''
# 例如，在 Agent/RuningMeanStd.py 文件中
import numpy as np


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon  # 样本计数器，初始化为小值防止除以零
        self.epsilon = epsilon  # 用于数值稳定的小偏置

    def update(self, x):
        """
        更新均值和方差的统计数据。
        x: 一个批次的数据，形状应为 (batch_size, state_dim)
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        根据批次的统计量更新全局统计量。
        """
        delta_mean = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta_mean * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        # M2 是更新后的平方差和
        M2 = m_a + m_b + np.square(delta_mean) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        """
        对输入数据进行归一化。
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)