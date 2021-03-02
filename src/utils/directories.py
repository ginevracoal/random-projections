DATA_PATH = "../../adversarial_examples/data/"
EXPERIMENTS = "../experiments/"

def _get_model_savedir(model_name, dataset_name, epochs, size_proj=None, projection_mode=None, centroid_translation=False):

    filepath = dataset_name+"_ep="+str(epochs)

    if size_proj:
        filepath += "_sizeproj="+str(size_proj)+"_"+str(projection_mode)

    if centroid_translation:
        filepath += "_centroid"

    filename = model_name+"_"+filepath+"_weights"

    return filename, EXPERIMENTS+filepath

    # def _set_baseline_filename(self, seed):
    #     """ Sets baseline filenames inside randens folder based on the projection seed. """
    #     filename = self.dataset_name + "_baseline" + "_size=" + str(self.size_proj) + \
    #                "_" + str(self.projection_mode)

    #     if self.epochs == None:
    #         filename = filename + "_" + str(seed)
    #     else:
    #         filename = filename + "_epochs=" + str(self.epochs) + "_" + str(seed)
    #     return filename


# def _get_attack_savedir():

#     return filename, filepath