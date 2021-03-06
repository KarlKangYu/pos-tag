from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import time

tf.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def do_inference(hostport):
    """Tests PredictionService with concurrent requests.
    Args:
    hostport: Host:port address of the Prediction Service.
    Returns:
    pred values, ground truth label
    """
    # create connection
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # initialize a request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'gec_classification_model'
    request.model_spec.signature_name = 'gec_classify'

    # Randomly generate some test data
    num_steps = 30
    # words = "1 27 10 272 5 54 48 219 11 360 53 17 628 1846 3"
    # tags = "11 14 7 8 2 4 3 6 16 7 11 22 13 10"
    # name_entity = "2 2 2 2 2 2 2 2 2 2 2 2 2 2"
    # words = words.strip().split()
    # words = [int(i) for i in words]
    # words = words + [0] * (num_steps - len(words))
    # tags = tags.strip().split()
    # tags = [int(i) for i in tags]
    # tags = tags + [0] * (num_steps - len(tags))
    # name_entity = name_entity.strip().split()
    # name_entity = [int(i) for i in name_entity]
    # name_entity = name_entity + [0] * (num_steps - len(name_entity))
    #
    # words = np.array([words])
    # tags = np.array([tags])
    # print(tags.shape)
    # name_entity = np.array([name_entity])

    words = [[3] * 30]
    tags = [[3] * 30]
    name_entity = [[2] * 30]


    request.inputs['input_data_words'].CopyFrom(
        tf.contrib.util.make_tensor_proto(words, shape=[1, num_steps]))
    request.inputs['input_data_tags'].CopyFrom(
        tf.contrib.util.make_tensor_proto(tags, shape=[1, num_steps]))
    request.inputs['input_data_name_entity'].CopyFrom(
        tf.contrib.util.make_tensor_proto(name_entity, shape=[1, num_steps]))

    # ##########################
    # request.inputs['input_dropout_keep_1'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(1.0, shape=[]))
    # request.inputs['input_dropout_keep_2'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(1.0, shape=[]))
    # request.inputs['input_tempreture'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(1.0, shape=[]))
    # request.inputs['input_training'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(False, shape=[]))
    # ##########################

    #
    # # Randomly generate some test data
    # temp_data = numpy.random.randn(10, 3).astype(numpy.float32)
    # data, label = temp_data, numpy.sum(temp_data * numpy.array([1, 2, 3]).astype(numpy.float32), 1)
    # request.inputs['input'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(data, shape=data.shape))

    # predict
    time_start = time.time()
    for i in range(10000):
        result = stub.Predict(request, 5.0)  # 5 seconds
    time_end = time.time()
    t = time_end - time_start
    return t


def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return

    result = do_inference(FLAGS.server)
    print('10000 times cost : ', result)


if __name__ == '__main__':
    tf.app.run()