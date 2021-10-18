
import tensorflow as tf
import tensorflow_federated as tff
import collections

'''
    FedSGD:本地迭代轮数E=1, 批量大小B=训练节点全部数据
'''

NUM_ROUNDS = 20
NUM_CLIENTS = 10
# BATCH_SIZE = 20  # 批量大小
# SHUFFLE_BUFFER = 100  # 数据元素每100个打乱
# PREFETCH_BUFFER = 10

# 加载数据
data_train, data_test = tff.simulation.datasets.emnist.load_data()
# 训练节点：固定选取训练节点，之后论文可以优化节点选择问题的地方
index_clients = data_train.client_ids[0:NUM_CLIENTS]

MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


def preprocess(dataset):
    def batch_format(element):
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1])
        )

    # return dataset.shuffle(SHUFFLE_BUFFER, seed=1).batch(
    #     BATCH_SIZE).map(batch_format).prefetch(PREFETCH_BUFFER)
    return dataset.map(batch_format)


# 准备n个训练节点的数据(已预处理)
def make_federated_data(clients_data, client_ids):
    return [preprocess(clients_data.create_tf_dataset_for_client(n))
            for n in client_ids]


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
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)
    # 正确结果
    flat_labels = tf.reshape(batch['y'], [-1])
    # 定义损失函数
    loss = -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
    # 定义准确度
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(predictions, flat_labels), tf.float32))
    # 计算样本个数
    num_examples = tf.cast(tf.size(batch['y']), tf.float32)
    # 累加本地模型训练结果，得到全局模型训练结果
    var.num_examples.assign_add(num_examples)
    var.loss_sum.assign_add(loss * num_examples)
    var.accuracy_sum.assign_add(accuracy * num_examples)

    return loss, predictions


# 获取本地模型的训练结果：损失函数，精确度，样本数量
def get_local_metrics(var):
    return collections.OrderedDict(
        num_examples=var.num_examples,
        loss=var.loss_sum / var.num_examples,
        accuracy=var.accuracy_sum / var.num_examples
    )


# 聚合算法FedSGD
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

    # 全局模型参数
    @property
    def local_variables(self):
        return [
            self._variables.num_examples,
            self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    # 定义输入输出的规格
    @property
    def input_spec(self):
        return collections.OrderedDict(
            x=tf.TensorSpec([None, 784], tf.float32),
            y=tf.TensorSpec([None, 1], tf.int32)
        )

    @tf.function
    def predict_on_batch(self, x, training=True):
        del training
        return predict_on_batch(self._variables, x)

    @tf.function
    def forward_pass(self, batch_input, training=True):
        del training
        loss, predictions = forward_pass(self._variables, batch_input)
        num_examples = tf.shape(batch_input['x'])[0]
        return tff.learning.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_examples
        )

    @tf.function
    def report_local_outputs(self):
        return get_local_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return global_aggregate


# 定义联邦训练过程
def federated_process():
    return tff.learning.build_federated_averaging_process(
        MnistModel,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    )


'''
    联邦学习
    :param num_rounds 联邦学习轮数
    :param iter_process 每轮训练过程
    :param federated_data 训练节点的数据
'''


def start_train(num_rounds, iter_process, federated_data):
    # tensorboard 日志文件
    logdir = "./tmp/logs/"
    summary_writer = tf.summary.create_file_writer(logdir)
    state = iter_process.initialize()
    with summary_writer.as_default():
        for round_i in range(1, num_rounds):
            state, metrics = iter_process.next(state, federated_data)
            print('round {:2d}, metrics={}'.format(round_i, metrics))
            for name, value in metrics['train'].items():
                tf.summary.scalar(name, value, step=round_i)


if __name__ == '__main__':
    # 准备训练数据
    federated_data_train = make_federated_data(data_train, index_clients)
    # 定义每一轮的联邦训练过程
    federated_iter_process = federated_process()
    # 开始联邦学习
    start_train(NUM_ROUNDS, federated_iter_process, federated_data_train)


