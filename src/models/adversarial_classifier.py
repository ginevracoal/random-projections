# -*- coding: utf-8 -*-

import os
import time
import random
import warnings
import numpy as np
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from keras.models import Model
from keras.layers import Input
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier as sklKerasClassifier

from utils import directories
from utils.data import save_to_pickle, load_from_pickle
from utils.directories import *
from utils.projection_functions import compute_distances


class AdversarialClassifier(sklKerasClassifier):
    """
    Adversarial Classifier base class
    """

    def __init__(self, input_shape, num_classes, data_format, dataset_name, epochs):


        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data_format = data_format
        self.dataset_name = dataset_name
        self.epochs = epochs
        # self.folder, self.filename = self._set_model_path().values()
        # self.attack_library = attack_library  # art, cleverhans
        self.trained = False
        
        self.classes_ = self._set_classes()
        self.model = self._set_model()

        super(AdversarialClassifier, self).__init__(build_fn=self.model, epochs=epochs)

    def _set_model_path(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _set_training_params(test, epochs):
        raise NotImplementedError

    def _get_logits(self, inputs):
        raise NotImplementedError

    @staticmethod
    def _set_session(device):
        """ Initialize tf session """
        # print(device_lib.list_local_devices())

        if device == "gpu":
            n_jobs = 1
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
            config.gpu_options.per_process_gpu_memory_fraction = 1 / n_jobs
            sess = tf.compat.v1.Session(config=config)
            keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
            sess.run(tf.global_variables_initializer())
            return sess
        elif device == "cpu":
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            sess.run(tf.global_variables_initializer())
            return sess

    def _set_model(self):
        """
        defines the layers structure for the classifier
        :return: model
        """
        inputs = Input(shape=self.input_shape)
        outputs = self._get_logits(inputs=inputs)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def _set_classes():
        """ Setting classes_ attribute for sklearn KerasClassifier class """
        return np.array(np.arange(10))

    def _set_device_name(self, device):
        if device == "gpu":
            return get_available_gpus()[0]
        elif device == "cpu":
            return "/CPU:0"
        else:
            raise AssertionError("Wrong device name.")

    def set_optimizer(self):
        return keras.optimizers.Adam()

    def train(self, x_train, y_train, device, batch_size=128):
        print("\nTraining infos:\nbatch_size = ", batch_size, "\nepochs = ", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        device_name = self._set_device_name(device)
        with tf.device(device_name):
            optimizer = self.set_optimizer()
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

            callbacks = []
            tensorboard = keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True,
                                                      write_images=True)
            es = keras.callbacks.EarlyStopping(monitor='loss', verbose=1)
            callbacks.append(es)

            # if self.epochs == None:
            #     # epochs = 50
            #     epochs = 20
            # else:
            #     epochs = self.epochs
            # if self.test == False:
            callbacks.append(tensorboard)

            start_time = time.time()
            self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=batch_size, callbacks=callbacks,
                           shuffle=True, validation_split=0.2)
            print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))
            self.trained = True

            return self

    def predict(self, x, **kwargs):
        return self.model.predict(x)
        # return np.argmax(self.model.predict(x), axis=1)

    def evaluate(self, x, y):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param x: test data
        :param y: test labels
        :return: predictions
        """
        if self.trained:
            classification_prob = self.predict(x)
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(classification_prob, axis=1)
            nb_correct_adv_pred = np.sum(y_pred == y_true)

            print("Correctly classified: {}".format(nb_correct_adv_pred))
            print("Incorrectly classified: {}".format(len(x) - nb_correct_adv_pred))

            acc = nb_correct_adv_pred / y.shape[0]
            print("Accuracy: %.2f%%" % (acc * 100))
            # print(classification_report(y_true, y_pred, labels=list(range(self.num_classes))))
            return classification_prob, y_true, y_pred
        else:
            raise AttributeError("Train your classifier before the evaluation.")

    @staticmethod
    def _get_norm(attack):
        """ Returns the norm used for computing perturbations on the given method. """
        return np.inf

    def generate_adversaries(self, x, y, attack_method, attack_library, device, eps=0.3):
        """
        Generates adversaries on the input data x using a given attack method.
        """
        random.seed(0)
        def batch_generate(attacker, x, batches=10):
            x_batches = np.split(x, batches)
            x_adv = []
            for idx, x_batch in enumerate(x_batches):
                x_adv.append(attacker.generate_np(x_val=x_batch))
            x_adv = np.vstack(x_adv)
            return x_adv

        x_adv = None

        if self.trained:
            print("\nGenerating adversaries with", attack_method, "method on", self.dataset_name)
            with warnings.catch_warnings():
                if attack_library == "art":
                    import art.attacks
                    from art.classifiers import KerasClassifier as artKerasClassifier
                    from art.utils import master_seed

                    classifier = artKerasClassifier(clip_values=(0,255), model=self.model)
                    # master_seed(0)

                    # classifier._loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
                    # classifier.custom_loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
                    if attack_method == 'fgsm':
                        attacker = art.attacks.FastGradientMethod(classifier, eps=0.3)
                        x_adv = attacker.generate(x=x)
                    elif attack_method == 'deepfool':
                        attacker = art.attacks.DeepFool(classifier, nb_grads=5)
                        x_adv = attacker.generate(x)
                    elif attack_method == 'virtual':
                        attacker = art.attacks.VirtualAdversarialMethod(classifier)
                        x_adv = attacker.generate(x)
                    elif attack_method == 'carlini':
                        attacker = art.attacks.CarliniLInfMethod(classifier, targeted=False, eps=0.5)
                        x_adv = attacker.generate(x=x)
                    elif attack_method == 'pgd':
                        attacker = art.attacks.ProjectedGradientDescent(classifier, eps=eps)
                        x_adv = attacker.generate(x=x)
                    elif attack_method == 'newtonfool':
                        attacker = art.attacks.NewtonFool(classifier, eta=0.3)
                        x_adv = attacker.generate(x=x)
                    elif attack_method == 'boundary':
                        attacker = art.attacks.BoundaryAttack(classifier, targeted=False, max_iter=500, delta=0.05)
                        # y = np.random.permutation(y)
                        x_adv = attacker.generate(x=x)
                    elif attack_method == 'spatial':
                        attacker = art.attacks.SpatialTransformation(classifier, max_translation=3.0, num_translations=5,
                                                         max_rotation=8.0,
                                                         num_rotations=3)
                        x_adv = attacker.generate(x=x)
                    elif attack_method == 'zoo':
                        attacker = art.attacks.ZooAttack(classifier)
                        x_adv = attacker.generate(x=x, y=y)
                    else:
                        raise("wrong attack name.")

                elif attack_library == "cleverhans":
                    import cleverhans.attacks
                    from cleverhans.utils_keras import KerasModelWrapper

                    session = self._set_session(device=device)
                    classifier = KerasModelWrapper(self.model)

                    if attack_method == 'fgsm':
                        attacker = cleverhans.attacks.FastGradientMethod(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack == 'deepfool':
                        attack_method = cleverhans.attacks.DeepFool(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack_method == 'carlini':
                        attacker = cleverhans.attacks.CarliniWagnerL2(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack_method == 'pgd':
                        attacker = cleverhans.attacks.ProjectedGradientDescent(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack_method == 'spatial':
                        attacker = cleverhans.attacks.SpatialTransformationMethod(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack_method == 'virtual':
                        attacker = cleverhans.attacks.VirtualAdversarialMethod(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack_method == 'saliency':
                        attacker = cleverhans.attacks.SaliencyMapMethod(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                else:
                    raise ValueError("wrong pkg name.")
        else:
            raise AttributeError("Train your classifier first.")

        print("Distance from perturbations: ", compute_distances(x, x_adv, ord=self._get_norm(attack_method)))
        return x_adv

    def save_adversaries(self, data, attack_method, attack_library, debug):
        """
        Save adversarially augmented test set.
        """
        filename, filepath = directories._get_attack_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                                                 epochs=self.epochs, debug=debug, 
                                                 attack_method=attack_method, attack_library=attack_library)
        save_to_pickle(data=data, filepath=filepath+"/", filename=filename)

    def load_adversaries(self, attack_method, attack_library, debug):

        filename, filepath = directories._get_attack_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                                                 epochs=self.epochs, debug=debug, 
                                                 attack_method=attack_method, attack_library=attack_library)
        return load_from_pickle(fullpath=filepath+"/"+filename)

    def save_classifier(self, filepath, filename):
        """
        Saves the trained model and adds the current datetime to the filepath.
        """
        filepath+="/"
        os.makedirs(filepath, exist_ok=True)
        fullpath = filepath + filename + ".h5"
        print("\nSaving classifier: ", fullpath)
        self.model.save(fullpath)

    def load_classifier(self, filepath, filename):
        """
        Loads a pre-trained classifier.
        """
        filepath+="/"
        print("\nLoading model: ", filepath + filename + ".h5")
        self.model = load_model(filepath + filename + ".h5")
        self.trained = True
        return self

