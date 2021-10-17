
import tensorflow as tf
import tensorflow_federated as tff
import collections

MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


# 定义训练模型的参数：w,b,num_exp,loss,accuracy（TensorFlow变量）
def create_mnist_variables():
    return MnistVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10,)),
            name='bias',
            trainable=True),
        num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
        loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
        accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


# 数据批量上的操作：先做wx+b的线性变换，再用softmax函数输出预测结果的概率形式
def predict_on_batch(var, x):
    return tf.nn.softmax(tf.matmul(x, var.weights) + var.bias)


# 前向传播
def forward_pass(var, batch):
    # 预测结果
    y = predict_on_batch(var, batch['x'])
    prediction = tf.cast(tf.argmax(y, 1), tf.int32)
    # 正确结果
    flat_labels = tf.reshape(batch['y'], [-1])
    # 定义损失函数
    loss = -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
    # 定义准确度
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(prediction, flat_labels), tf.float32))
    # 计算样本个数
    num_examples = tf.cast(tf.size(batch['y']), tf.float32)
    # 记录全局模型参数
    var.num_examples.assign_add(num_examples)
    var.loss_sum.assign_add(loss * num_examples)
    var.accuracy_sum.assign_add(num_examples * num_examples)

    return loss, prediction


# 获取本地模型的训练结果：损失函数，精确度，样本数量(用于FedAVG加权平均)
def get_local_metrics(var):
    return collections.OrderedDict(
        num_examples=var.num_examples,
        loss=var.loss_sum / var.num_examples,
        accuracy=var.accuracy_sum / var.num_examples
    )


# 聚合算法FedAVG，之后可以在此处自定义聚合算法
@tff.federated_computation
def global_aggregate(metrics):
    return collections.OrderedDict(
        num_all_examples=tff.federated_sum(metrics.num_examples),
        loss=tff.federated_mean(metrics.loss, metrics.num_examples),
        accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples)
    )


# 自定义用于Mnist数据集的联邦学习训练模型（实现tff.learning.Model接口）
class MnistModel(tff.learning.Model):

    def __init__(self):
        self._variables = create_mnist_variables()

    @property
    def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples,
            self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        pass

    def forward_pass(self, batch_input, training=True):
        pass

    def report_local_outputs(self):
        pass

    def federated_output_computation(self):
        pass


