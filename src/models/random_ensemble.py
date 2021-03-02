# -*- coding: utf-8 -*-

"""
This model computes random projections of the input points in a lower dimensional space and performs classification
separately on each projection, then it returns an ensemble classification on the original input data.
"""


from utils import directories
from models.baseline_convnet import *
from utils.projection_functions import *
# from utils.robustness_measures import softmax_difference

############
# defaults #
############

REPORT_PROJECTIONS = False

class RandomEnsemble(BaselineConvnet):
    """
    Classifies `n_proj` random projections of the training data in a lower dimensional space (whose dimension is
    `size_proj`^2), then classifies the original high dimensional data with an ensemble classifier, summing up the
    probabilities from the single projections.
    """
    def __init__(self, input_shape, num_classes, n_proj, size_proj, projection_mode, data_format, dataset_name, 
                 attack_library, epochs=None, centroid_translation=False):
        """
        Extends BaselineConvnet initializer with additional informations about the projections.

        :param input_shape:
        :param num_classes:
        :param n_proj: number of random projections
        :param size_proj: size of a random projection
        :param projection_mode: method for computing projections on RGB images
        :param data_format: channels first or last
        :param dataset_name:
        :param test: if True only takes the first 100 samples
        """

        if size_proj > input_shape[1]:
            raise ValueError("The size of projections has to be lower than the image size.")

        super(RandomEnsemble, self).__init__(input_shape, num_classes, data_format, dataset_name, epochs)

        self.model_name="randens"
        self.centroid_translation = centroid_translation # todo: refactor centroid translation
        self.translation_vector = None
        self.original_input_shape = input_shape
        self.n_proj = n_proj
        self.size_proj = size_proj
        self.projection_mode = projection_mode
        self.random_seeds = range(0,n_proj)
        self.input_shape = (size_proj, size_proj, input_shape[2])
        self.classifiers = None
        self.training_time = 0
        self.ensemble_method = "sum"  # supported methods: mode, sum
        self.x_test_proj = None
        self.baseline_classifier = None

        print("\n === RandEns model ( n_proj = ", self.n_proj, ", size_proj = ", self.size_proj, ") ===")

    def compute_projections(self, input_data, translation):
        """ Extends utils.compute_projections method in order to handle the third input dimension."""
        projections, inverse_projections = compute_projections(input_data, self.random_seeds, self.n_proj,
                                                               self.size_proj, self.projection_mode,
                                                               translation=translation)

        # eventually adjust input dimension to a single channel projection
        if projections.shape[4] == 1:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 1)
        else:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 3)

        return projections, inverse_projections

    def _set_translation_vector(self, x_train):
        if self.centroid_translation:
            return np.mean(np.mean(x_train, axis=0), axis=2)
        else:
            return None
            # return np.zeros(shape=[1, rows, cols, channels], dtype=np.float32)

    def train(self, x_train, y_train, device):
        """
        Trains the baseline model over `n_proj` random projections of the training data whose input shape is
        `(size_proj, size_proj, 1)`.

        :param x_train: training data
        :param y_train: training labels
        :return: list of n_proj trained models, which are art.KerasClassifier fitted objects
        """
        self.translation_vector = self._set_translation_vector(x_train)
        device_name = self._set_device_name(device)

        with tf.device(device_name):
            start_time = time.time()
            input_data = x_train.astype(float)
            x_train_projected, _ = self.compute_projections(input_data=input_data,
                                                            translation=self.translation_vector)

            # eventually adjust input dimension to a single channel projection
            if x_train_projected.shape[4] == 1:
                self.input_shape = (self.input_shape[0],self.input_shape[1],1)

            classifiers = []
            for i in self.random_seeds:
                # use the same model architecture (not weights) for all trainings
                baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                           data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
                # train n_proj classifiers on different training data
                classifiers.append(baseline.train(x_train_projected[i], y_train, device))
                del baseline

            print("\nTraining time for model ( n_proj =", str(self.n_proj), ", size_proj =", str(self.size_proj),
                  "): --- %s seconds ---" % (time.time() - start_time))

            self.trained = True
            self.classifiers = classifiers
            return classifiers

    def _sum_ensemble_classifier(self, classifiers, projected_data):
        """
        :param classifiers: list of `n_proj` different GaussianRandomProjection objects
        :param projected_data: array of test data projected on the different `n_proj` training random directions
        (`size_proj` directions for each projection)
        :return: sum of all the predicted probabilities among each class for the `n_proj` classifiers

        ```
        Predictions on the first element by each single classifier
        predictions[:, 0] =
         [[0.04745529 0.00188083 0.01035858 0.21188359 0.00125252 0.44483757
          0.00074033 0.13916749 0.01394993 0.1284739 ]
         [0.00259137 0.00002327 0.48114488 0.42658636 0.00003032 0.01012747
          0.00002206 0.03735029 0.02623402 0.0158899 ]
         [0.00000004 0.00000041 0.00000277 0.00001737 0.00000067 0.00000228
          0.         0.9995009  0.00000014 0.0004754 ]]

        Ensemble prediction vector on the first element
        summed_predictions[0] =
         [0.05004669 0.00190451 0.49150622 0.6384873  0.0012835  0.45496735
         0.0007624  1.1760187  0.04018409 0.14483918]
        """
        # compute predictions for each projection
        proj_predictions = np.array([classifier.predict(projected_data[i]) for i, classifier in enumerate(classifiers)])

        # todo: little analysis on ensemble behaviour
        # print(proj_predictions[:,0,:])

        # sum the probabilities across all predictors
        predictions = np.sum(proj_predictions, axis=0)
        return predictions

    def _mode_ensemble_classifier(self, classifiers, projected_data):
        """
        :param classifiers: list of `n_proj` different GaussianRandomProjection objects
        :param projected_data: array of test data projected on the different `n_proj` training random directions
        (`size_proj` directions for each projection)
        :return: compute the argmax of the probability vectors and then, for each points, choose the mode over all
        classes as the predicted label

        ```
        Computing random projections.
        Input shape:  (100, 28, 28, 1)
        Projected data shape: (3, 100, 8, 8, 1)

        Predictions on the first element by each single classifier
        predictions[:, 0] =
        [[0.09603461 0.08185963 0.03264992 0.07047556 0.2478332  0.03418195
          0.13880958 0.19712913 0.04649669 0.05452974]
         [0.0687536  0.14464664 0.0766349  0.09082405 0.1066305  0.01555605
          0.03265413 0.12625733 0.14203466 0.19600812]
         [0.16379683 0.0895557  0.07057846 0.09945401 0.25141633 0.04555665
          0.08481159 0.0610559  0.06158736 0.07218721]]

        argmax_predictions[:, 0] =
        [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]

        Ensemble prediction vector on the first element
        summed_predictions[0] =
        [0.         0.         0.         0.         0.66666667 0.
         0.         0.         0.         0.33333333]
        ```
        """

        # compute predictions for each projection
        proj_predictions = np.array([classifier.predict(projected_data[i]) for i, classifier in enumerate(classifiers)])

        # convert probability vectors into target vectors
        argmax_predictions = np.zeros(proj_predictions.shape)
        for i in range(proj_predictions.shape[0]):
            idx = np.argmax(proj_predictions[i], axis=-1)
            argmax_predictions[i, np.arange(proj_predictions.shape[1]), idx] = 1
        # sum the probabilities across all predictors
        predictions = np.sum(argmax_predictions, axis=0)
        # normalize
        predictions = predictions / predictions.sum(axis=1)[:, None]
        return predictions

    def predict(self, x, add_baseline_prob=False):
        """
        Compute the ensemble prediction probability vector by summing up the probability vectors obtained over different
        projections.
        :param x: input data
        :param add_baseline_prob: if True adds baseline probabilities to logits layer
        :return: probability vector final predictions on x
        """
        projected_data, _ = self.compute_projections(x, translation=self.translation_vector)
        predictions = None
        if self.ensemble_method == 'sum':
            predictions = self._sum_ensemble_classifier(self.classifiers, projected_data)
        elif self.ensemble_method == 'mode':
            predictions = self._mode_ensemble_classifier(self.classifiers, projected_data)

        if add_baseline_prob:
            print("\nAdding baseline probability vector to the predictions.")
            baseline = BaselineConvnet(input_shape=self.original_input_shape, num_classes=self.num_classes,
                                       data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
            baseline.load_classifier(relative_path=TRAINED_MODELS, folder="baseline/", filename=baseline.filename)
            baseline_predictions = baseline.predict(x)

            # sum the probabilities across all predictors
            final_predictions = np.add(predictions, baseline_predictions)
            return final_predictions
        else:
            return predictions

    def report_projections(self, classifiers, x_test_proj, y_test):
        """
        Computes classification reports on each projection.
        """
        print("\n === projections report ===")
        # proj_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
        #                            data_format=self.data_format, dataset_name=self.dataset_name, test=True)
        for i, proj_classifier in enumerate(classifiers):
            print("\nTest evaluation on projection ", self.random_seeds[i])
            proj_classifier.evaluate(x=x_test_proj[i], y=y_test)

    def evaluate(self, x, y, report_projections=False):
        """ Extends evaluate() with projections reports"""
        classification_prob, y_true, y_pred = super(RandomEnsemble, self).evaluate(x, y)
        # y_pred = self.classifiers.evaluate(x, y)
        if report_projections:
            x_proj, _ = self.compute_projections(x, translation=self.translation_vector)
            self.report_projections(classifiers=self.classifiers, x_test_proj=x_proj, y_test=y)
        # print(classification_prob[0],y_true[0],y_pred[0])
        return classification_prob, y_true, y_pred

    def generate_adversaries(self, x, y, attack, seed=0, eps=None, device="cpu"):
        """ Adversaries are generated on the baseline classifier """

        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                   data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
        baseline.load_classifier(relative_path=TRAINED_MODELS + MODEL_NAME + "/")
        x_adv = baseline.generate_adversaries(x, y, attack=attack, eps=eps)
        return x_adv

    # def _set_model_path(self, model_name="randens"):
    #     folder = MODEL_NAME + "/" + self.dataset_name + "_" + str(model_name) + "_size=" + str(self.size_proj) +\
    #              "_" + str(self.projection_mode)
    #     if self.centroid_translation:
    #         folder = folder + "_centroid"
    #     return {'folder': folder + "/", 'filename': None}

    # def _set_baseline_filename(self, seed):
    #     """ Sets baseline filenames inside randens folder based on the projection seed. """
    #     filename = self.dataset_name + "_baseline" + "_size=" + str(self.size_proj) + \
    #                "_" + str(self.projection_mode)
    #     if self.centroid_translation:
    #         filename = filename + "_centroid"

    #     if self.epochs == None:
    #         filename = filename + "_" + str(seed)
    #     else:
    #         filename = filename + "_epochs=" + str(self.epochs) + "_" + str(seed)
    #     return filename

    def save_classifier(self, filename=None, filepath=None):
        """
        Saves all projections classifiers separately.
        :param relative_path: relative path of the folder containing the list of trained classifiers.
                              It can be either TRAINED_MODELS or RESULTS
        :param filename: filename
        """

        filename, filepath = directories._get_model_savedir(model_name=self.model_name, dataset_name=self.dataset_name, 
                        size_proj=self.size_proj, projection_mode=self.projection_mode, epochs=self.epochs, seed=None, 
                        centroid_translation=False)

        if self.trained:
            for seed, proj_classifier in enumerate(self.classifiers):

                filename += "_"+str(seed)
                proj_classifier.save_classifier(relative_path=relative_path, folder=self.folder,
                                                filename=filename)
            if self.centroid_translation:
                save_to_pickle(data=self.translation_vector,
                               relative_path="../results/" + str(time.strftime('%Y-%m-%d')) + "/" + self.folder,
                               filename="training_data_centroid.pkl")
        else:
            raise ValueError("Train the model first.")

    def load_classifier(self, relative_path, folder=None, filename=None):
        """
        Loads a pre-trained classifier and sets the projector with the training seed.
        :param relative_path: relative path of the folder containing the list of trained classifiers.
                              It can be either TRAINED_MODELS or RESULTS
        :param filename: filename
        :return: list of trained classifiers
        """
        start_time = time.time()
        self.trained = True

        classifiers = []
        for i in self.random_seeds:
            proj_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, test=self.test,
                                              data_format=self.data_format, dataset_name=self.dataset_name)
            classifiers.append(proj_classifier.load_classifier(relative_path=relative_path, folder=self.folder,
                                                               filename=self._set_baseline_filename(seed=i)))
        if self.centroid_translation:
            self.translation_vector = load_from_pickle(path=relative_path + self.folder + "training_data_centroid.pkl",
                                                       test=False)
        else:
            self.translation_vector = None

        print("\nLoading time: --- %s seconds ---" % (time.time() - start_time))

        self.classifiers = classifiers
        return classifiers
