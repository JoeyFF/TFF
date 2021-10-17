import tensorflow as tf
import tensorflow_federated as tff
import collections

'''
    @author:ZB404
    TFF框架+Keras
    目前只使用FL API实现MNIST数据集图像分类，
    之后会使用FC API实现自定义的联邦学习场景（包括所用机器学习模型、聚合算法、节点选择）
    环境：python_3.6，tensorflow_2.5.1，tensorflow-federated_0.19.0
'''

# 定义训练常量
NUM_CLIENTS = 10  # 训练节点数
NUM_EPOCHS = 5  # 本地训练轮数, 以后论文可以优化全局聚合频率问题的地方
NUM_ROUNDS = 10  # 联邦学习轮数
BATCH_SIZE = 20  # 批量大小
SHUFFLE_BUFFER = 100  # 数据元素每100个打乱
PREFETCH_BUFFER = 10



# 加载数据
data_train, data_test = tff.simulation.datasets.emnist.load_data()
# 训练节点：固定选取训练节点，之后论文可以优化节点选择问题的地方
index_clients = data_train.client_ids[0:NUM_CLIENTS]

'''
    预处理数据
    将28*28的数字图像转换成长度为784的一维数组
    将输入输出的名称'pixels'和'label'改为'x'和'y'
    将数据打乱顺序并按照训练轮数重复
'''


def preprocess(dataset):
    def batch_format(element):
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1])
        )

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).map(batch_format).prefetch(PREFETCH_BUFFER)


# 准备n个训练节点的数据(已预处理)
def make_federated_data(clients_data, client_ids):
    return [preprocess(clients_data.create_tf_dataset_for_client(n))
            for n in client_ids]


# 返回训练节点的数据规格
def client_data_spec(clients_data):
    exp_client_data = clients_data.create_tf_dataset_for_client(clients_data.client_ids[0])
    return preprocess(exp_client_data).element_spec


# 创建keras模型：之后可使用自定义机器学习模型，只要实现tff.learning.Model接口
def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax()
    ])


# 创建训练模型
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=client_data_spec(data_train),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )


# 定义联邦训练过程:本地训练+全局聚合FedAVG
def federated_process():
    return tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
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
            for name, value in metrics['train'].items():
                tf.summary.scalar(name, value, step=round_i)


if __name__ == '__main__':
    # 准备训练数据
    federated_data_train = make_federated_data(data_train, index_clients)
    # 定义每一轮的联邦训练过程
    federated_iter_process = federated_process()
    # 开始联邦学习
    start_train(NUM_ROUNDS, federated_iter_process, federated_data_train)

