DATA_PATH = "../../adversarial_examples/data/"
EXPERIMENTS = "../experiments/"
LOG_DIR = EXPERIMENTS+"tensorboard/"

def _get_model_savedir(model_name, dataset_name, epochs, debug, 
                       size_proj=None, projection_mode=None, centroid_translation=False,
                       robust=False, attack_method=None, attack_library=None):

    filepath = model_name+"_"+dataset_name+"_ep="+str(epochs)

    if size_proj:
        filepath += "_size="+str(size_proj)+"_"+projection_mode

    if centroid_translation:
        filepath += "_centroid"

    filename = filepath

    if robust:
        filename += "_"+attack_library+"_"+attack_method+"_robust"

    filename += "_weights"

    filepath = EXPERIMENTS+"debug/"+filepath if debug else EXPERIMENTS+filepath

    return filename, filepath+"/"


def _get_attack_savedir(model_name, dataset_name, epochs, debug, attack_method, attack_library,
                        size_proj=None, projection_mode=None, centroid_translation=False, robust=False):

    _, filepath = _get_model_savedir(model_name=model_name, dataset_name=dataset_name, epochs=epochs, debug=debug, 
                           size_proj=size_proj, projection_mode=projection_mode, centroid_translation=centroid_translation,
                           robust=robust, attack_method=attack_method, attack_library=attack_library)

    filename = attack_library+"_"+attack_method+"_attack"

    return filename, filepath