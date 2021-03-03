import random
from utils.data import *
from utils.directories import *
from tensorflow.python.client import device_lib

DATASETS = "mnist, cifar"
ATTACKS = "fgsm, pgd, deepfool, carlini, newtonfool, virtual"
PROJ_MODE = "flat, channels, one_channel, grayscale"

random.seed(0)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def set_session(device, n_jobs):
    """
    Initialize tf session on device.
    """

    # from keras.backend.tensorflow_backend import set_session
    sess = tf.Session()
    # print(device_lib.list_local_devices())
    if device == "gpu":
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        # config.allow_soft_placement = True
        # config.log_device_placement = True  # to log device placement (on which device the operation ran)
        config.gpu_options.per_process_gpu_memory_fraction = 1/n_jobs
        sess = tf.Session(config=config)

    keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
    # sess.run(tf.compat.v1.global_variables_initializer())
    # print("check cuda: ", tf.test.is_built_with_cuda())
    # print("check gpu: ", tf.test.is_gpu_available())
    return sess

def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))

