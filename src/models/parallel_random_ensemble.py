# -*- coding: utf-8 -*-

import keras
from keras import backend as K
from models.random_ensemble import *
from utils.projection_functions import compute_single_projection
from utils.data import load_dataset
from joblib import Parallel, delayed


class ParallelRandomEnsemble(RandomEnsemble):

    def __init__(self, input_shape, num_classes, size_proj, n_proj, data_format, dataset_name, #proj_idx, 
                 projection_mode, epochs, n_jobs, centroid_translation=False):
        super(ParallelRandomEnsemble, self).__init__(input_shape=input_shape, num_classes=num_classes, n_proj=n_proj,
                                                     size_proj=size_proj, projection_mode=projection_mode,
                                                     data_format=data_format, dataset_name=dataset_name, 
                                                     epochs=epochs, centroid_translation=centroid_translation)
        # self.proj_idx = proj_idx
        self.n_jobs = n_jobs
        # _set_session(device, n_jobs=self.n_jobs)

    @staticmethod
    def _set_session(device):
        K.clear_session()
        return tf.reset_default_graph()

    def train_single_projection(self, x_train, y_train, device, proj_idx, debug):
        """ Trains a single projection of the ensemble classifier and saves the model in current day results folder."""

        print("\nTraining single randens projection with seed =", str(proj_idx),
              "and size_proj =", str(self.size_proj))

        start_time = time.time()

        self.translation_vector = self._set_translation_vector(x_train)
        x_train_projected, x_train_inverse_projected = compute_single_projection(input_data=x_train,
                                                                                 proj_idx=proj_idx,
                                                                                 size_proj=self.size_proj,
                                                                                 projection_mode=self.projection_mode,
                                                                                 translation=self.translation_vector)
        # eventually adjust input dimension to a single channel projection
        if x_train_projected.shape[3] == 1:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 1)

        # use the same model architecture (not weights) for all trainings
        proj_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, epochs=self.epochs,
                                          data_format=self.data_format, dataset_name=self.dataset_name)
        proj_classifier.train(x_train_projected, y_train, device=device)
        print("\nProjection + training time: --- %s seconds ---" % (time.time() - start_time))
        self.trained = True

        filename, filepath = directories._get_model_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                                size_proj=self.size_proj, projection_mode=self.projection_mode, epochs=self.epochs,
                                centroid_translation=self.centroid_translation, debug=debug)

        proj_classifier.save_classifier(filepath=filepath, filename=filename+"_"+str(proj_idx), debug=debug)
        return proj_classifier

    def train(self, x_train, y_train, debug, device="cpu"):
        self._set_session(device)
        self.translation_vector = self._set_translation_vector(x_train)
        if self.centroid_translation:
                save_to_pickle(data=self.translation_vector, 
                               filepath=filepath, filename="training_data_centroid.pkl")

        Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_train)(x_train=x_train, y_train=y_train, dataset_name=self.dataset_name,
                                     input_shape=self.input_shape, num_classes=self.num_classes, 
                                     data_format=self.data_format, debug=debug, epochs=self.epochs,
                                     proj_idx=proj_idx, device=device, size_proj=self.size_proj,
                                     proj_mode=self.projection_mode, n_jobs=self.n_jobs, 
                                     centroid_translation=self.centroid_translation)
            for proj_idx in range(0, self.n_proj))

        self.trained = True
        # self.classifiers = classifiers
        # return classifiers

    # def compute_projections(self, input_data):
    #     """
    #     Parallel implementation of this method
    #     :param input_data:
    #     :return:
    #     """
    #     n_jobs = self.n_proj # 2 if device == "gpu" else self.n_proj
    #     projections = Parallel(n_jobs=n_jobs)(
    #         delayed(_parallel_compute_projections)(input_data, proj_idx=proj_idx, size_proj=self.size_proj,
    #                                                projection_mode=self.projection_mode, n_jobs=n_jobs,
    #                                                translation=self.translation)
    #         for proj_idx in self.random_seeds)
    #
    #     # eventually adjust input dimension to a single channel projection
    #     projections = np.array(projections)
    #
    #     if projections.shape[4] == 1:
    #         self.input_shape = (self.input_shape[0], self.input_shape[1], 1)
    #     else:
    #         self.input_shape = (self.input_shape[0], self.input_shape[1], 3)
    #     return projections, None

    def _sum_ensemble_classifier(self, classifiers, projected_data):
        """
        Parallelized version of this method.
        :param classifiers: list of trained classifiers
        :param projected_data: list of projected data for all of the n_proj random initializations
        :return:
        """
        # compute predictions for each projection
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_predict)(classifier=classifier, projected_data=projected_data[i], n_jobs=self.n_jobs)
            for i, classifier in enumerate(classifiers))
        proj_predictions = np.array(results)
        # sum the probabilities across all predictors
        predictions = np.sum(proj_predictions, axis=0)
        return predictions

    def load_classifier(self, debug, filepath=None, filename=None):
        n_jobs = self.n_proj
        K.clear_session()
        self._set_session("cpu")
        start_time = time.time()

        if filename is None or filepath is None:
            filename, filepath = directories._get_model_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                            size_proj=self.size_proj, projection_mode=self.projection_mode, epochs=self.epochs,
                            centroid_translation=self.centroid_translation, debug=debug)

        classifiers = Parallel(n_jobs=n_jobs)(
            delayed(_parallel_load_classifier)(input_shape=self.input_shape, num_classes=self.num_classes,
                                               data_format=self.data_format, dataset_name=self.dataset_name,
                                               filepath=filepath, filename=filename+"_"+str(i), n_jobs=n_jobs,
                                               debug=debug, epochs=self.epochs)
            for i in list(self.random_seeds))
        print("\nLoading time: --- %s seconds ---" % (time.time() - start_time))
        # for classifier in classifiers:
        #     classifier.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
        #                            metrics=['accuracy'])
        #     # classifier.model._make_predict_function()
        self.classifiers = classifiers
        if self.centroid_translation:
            self.translation_vector = load_from_pickle(path=relative_path + self.folder + "training_data_centroid.pkl",
                                                       test=False)
        return classifiers

    def evaluate(self, x, y, debug, report_projections=False, device="cpu", add_baseline_prob=False):
        """
        Computes parallel evaluation of the model, then joins the results from the single workers into the final
        probability vector.
        :param x: Input data
        :param y: Input labels
        :param report_projections: include classification labels (True/False)
        :param model_path: model path for loading (RESULTS/TRAINED_MODELS)
        :param device: model evaluation device (True/False)
        :return: accuracy on the predictions
        """
        K.clear_session()
        self._set_session(device)
        if self.centroid_translation:
            translation = load_from_pickle(path=model_path + self.folder + "training_data_centroid.pkl",
                                                       test=False)
        else:
            translation = None

        filename, filepath = directories._get_model_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                        size_proj=self.size_proj, projection_mode=self.projection_mode, epochs=self.epochs,
                        centroid_translation=self.centroid_translation, debug=debug)

        proj_predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_evaluate)(input_shape=self.input_shape, num_classes=self.num_classes, debug=debug,
                                        data_format=self.data_format, dataset_name=self.dataset_name,
                                        filepath=filepath, input_data=x, proj_idx=i, 
                                        filename=filename+"_"+str(i), size_proj=self.size_proj,
                                        projection_mode=self.projection_mode, translation=translation)
            for i in list(self.random_seeds))

        predictions = np.sum(np.array(proj_predictions), axis=0)

        if add_baseline_prob:
            print("\nAdding baseline probability vector to the predictions.")

            if filename is None or filepath is None:
                filename, filepath = directories._get_model_savedir(model_name="baseline", 
                                dataset_name=self.dataset_name, epochs=self.epochs, debug=debug)

            baseline = BaselineConvnet(input_shape=self.original_input_shape, num_classes=self.num_classes,
                                       data_format=self.data_format, dataset_name=self.dataset_name, epochs=self.epochs)
            baseline.load_classifier(filename=filename, filepath=filepath, debug=debug)
            baseline_predictions = baseline.predict(x)

            # sum the probabilities across all predictors
            predictions = np.add(predictions, baseline_predictions)

        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y, axis=1)
        nb_correct_adv_pred = np.sum(y_pred == y_true)

        print("\nCorrectly classified: {}".format(nb_correct_adv_pred))
        print("Incorrectly classified: {}".format(len(x) - nb_correct_adv_pred))

        acc = nb_correct_adv_pred / y.shape[0]
        print("Accuracy: %.2f%%" % (acc * 100))
        return acc

def _parallel_evaluate(input_shape, num_classes, debug, data_format, dataset_name, filepath, filename,
                       size_proj, input_data, proj_idx, projection_mode, translation):
    """ Parallel evaluation on single projections using BaselineConvnet base class. """
    classifier = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                dataset_name=dataset_name, epochs=epochs)
    
    classifier.load_classifier(debug=debug, filename=filename, filepath=filepath)
    print("Parallel computing projection ", proj_idx)
    projection, _ = compute_single_projection(input_data=input_data, proj_idx=proj_idx, size_proj=size_proj,
                                              projection_mode=projection_mode, translation=translation)
    prediction = classifier.predict(projection)
    return prediction


def _parallel_predict(classifier, projected_data, device="cpu"):
    # import tensorflow as tf
    # _set_session(device, n_jobs)
    # use the same computational graph of training for the predictions
    # g = tf.get_default_graph()
    # with g.as_default():
    predictions = classifier.predict(projected_data)
    return predictions


def _parallel_train(x_train, y_train, input_shape, epochs, num_classes, data_format, dataset_name, debug, 
                    proj_idx, size_proj, proj_mode, device, centroid_translation, n_jobs):
    print("\nParallel training projection ", proj_idx)
    # import tensorflow as tf
    # g = tf.get_default_graph()
    # _set_session(device, n_jobs)
    # with g.as_default():
    model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
                                   data_format=data_format, dataset_name=dataset_name, n_jobs=n_jobs,
                                   projection_mode=proj_mode, n_proj=1, epochs=epochs,
                                   centroid_translation=centroid_translation)

    model.train_single_projection(x_train=x_train, y_train=y_train, device=device, proj_idx=proj_idx, debug=debug)
    
#
# def _parallel_compute_projections(input_data, proj_idx, size_proj, projection_mode, n_jobs, translation):
#     _set_session(device, n_jobs)
#     print("\nParallel computing projection ", proj_idx)
#     projection, _ = compute_single_projection(input_data=input_data, seed=proj_idx, size_proj=size_proj,
#                                               projection_mode=projection_mode, translation=translation)
#     return projection


def _parallel_load_classifier(input_shape, num_classes, data_format, dataset_name, epochs, filepath, filename,
                              n_jobs, debug):
    # _set_session(device, n_jobs)
    classifier = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, epochs=epochs, 
                                 data_format=data_format, dataset_name=dataset_name)
    classifier.load_classifier(debug=debug, filename=filename, filepath=filepath)
    return classifier

