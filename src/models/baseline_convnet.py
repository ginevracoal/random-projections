# -*- coding: utf-8 -*-

"""
CNN model.
"""
import time
import copy
import random
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from utils import directories
from models.adversarial_classifier import AdversarialClassifier


class BaselineConvnet(AdversarialClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, epochs):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        super(BaselineConvnet, self).__init__(input_shape, num_classes, data_format, dataset_name, epochs)
        self.model_name = "baseline"

    def _get_logits(self, inputs):
        """
        Builds model architecture and returns logits layer
        :param inputs: input data
        :return: logits
        """
        inputs = tf.cast(inputs, tf.float32)
        if self.dataset_name == "mnist":
            x = Conv2D(32, kernel_size=(3, 3), activation='relu', data_format=self.data_format)(inputs)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            logits = Dense(self.num_classes, activation='softmax')(x)
            return logits

        elif self.dataset_name == "cifar":
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.2)(x)
            x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.3)(x)
            x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.4)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            logits = Dense(10, activation='softmax')(x)
            return logits

    def adversarial_train(self, x_train, y_train, device, attack_method, attack_library, seed=0):
        """
        Performs adversarial training on the given classifier using an attack method. Training set adversaries are
        generated at training time on the baseline model.
        :param x_train: training data
        :param y_train: training labels
        :param attack: adversarial attack
        :param seed: seed used in baseline model training
        :return: adversarially trained classifier
        """
        random.seed(seed)
        if self.trained:
            robust_classifier = copy.deepcopy(self) #.load_classifier(relative_path=TRAINED_MODELS)
        else:
            robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                                data_format=self.data_format, dataset_name=self.dataset_name)
            robust_classifier.train(x_train, y_train, device)

        start_time = time.time()
        print("\n===== Adversarial training =====")
        x_train_adv = self.generate_adversaries(x_train, y_train, attack_method=attack_method, 
                                                attack_library=attack_library, device=device)
        # x_train_adv = self.load_adversaries(attack=attack, relative_path=DATA_PATH)

        # Data augmentation: expand the training set with the adversarial samples
        # x_train_ext = np.append(x_train, x_train_adv, axis=0)
        # y_train_ext = np.append(y_train, y_train, axis=0)

        # Retrain the CNN on the extended dataset
        # robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, test=self.test,
        #                                     data_format=self.data_format, dataset_name=self.dataset_name)
        robust_classifier.train(x_train_adv, y_train, device)
        # robust_classifier.filename = self._robust_classifier_name(attack=attack, seed=seed)

        print("Adversarial training time: --- %s seconds ---" % (time.time() - start_time))
        return robust_classifier

    def save_classifier(self, debug, filename=None, filepath=None):

        if filename is None or filepath is None:
            filename, filepath = directories._get_model_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                                                                epochs=self.epochs, debug=debug)

        super(BaselineConvnet, self).save_classifier(filepath=filepath, filename=filename)

    def load_classifier(self, debug, filename=None, filepath=None):

        if filename is None or filepath is None:
            filename, filepath = directories._get_model_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                                                                epochs=self.epochs, debug=debug)

        super(BaselineConvnet, self).load_classifier(filepath=filepath, filename=filename)

    def save_robust_classifier(self, robust_classifier, debug, attack_method, attack_library, filename=None, filepath=None):

        if filename is None or filepath is None:
            filename, filepath = directories._get_model_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                                                                epochs=self.epochs, debug=debug, robust=True, 
                                                                attack_method=attack_method, attack_library=attack_library)

        robust_classifier.save_classifier(filepath=filepath, filename=filename, debug=debug)

    def load_robust_classifier(self, debug, attack_method, attack_library, filename=None, filepath=None):

        if filename is None or filepath is None:
            filename, filepath = directories._get_model_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                                                                epochs=self.epochs, debug=debug, robust=True, 
                                                                attack_method=attack_method, attack_library=attack_library)

        robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, 
                                            data_format=self.data_format, dataset_name=self.dataset_name)

        robust_classifier.load_classifier(filepath=filepath, filename=filename, debug=debug)
        return robust_classifier

    def train_const_SGD(self, x_train, y_train, device, epochs, lr):
        """
        Perform SGD optimization with constant learning rate on a pre-trained network.
        :param x_train: training data
        :param y_train: training labels
        :param epochs: number of epochs
        :param lr: learning rate
        :return: re-trained network
        """
        if self.trained:
            self.filename = "SGD_lr=" + str(lr) + "_ep=" + str(epochs) + "_"+ self.dataset_name + "_baseline.h5"
            self.epochs = epochs
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                                optimizer=keras.optimizers.SGD(lr=lr, clipnorm=1.),
                                metrics=['accuracy'])
            self.train(x_train=x_train, y_train=y_train, device=device)
            return self
        else:
            raise AttributeError("Train your classifier first.")

# def plot_attacks(dataset_name, epochs, debug, attacks, attack_library):

#     x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
#                                                                                            debug=debug)
#     model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
#                             dataset_name=dataset_name, epochs=epochs)
#     model.load_classifier(relative_path=RESULTS, filename=model.filename+"_seed="+str(seed))
#     images = []
#     labels = []
#     for attack in attacks:
#         eps = model._get_attack_eps(dataset_name=model.dataset_name, attack=attack)
#         x_test_adv = model.generate_adversaries(x=x_test, y=y_test, attack=attack, eps=eps)
#         images.append(x_test_adv)
#         avg_dist = compute_distances(x_test, x_test_adv, ord=model._get_norm(attack))['mean']
#         labels.append(str(attack) + " avg_dist=" + str(avg_dist))

#     plot_images(image_data_list=images,labels=labels)

