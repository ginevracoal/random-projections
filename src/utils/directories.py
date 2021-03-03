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

    if robust:
        filepath += attack_library+"_"+attack_method+"_robust"

    filename = filepath+"_weights"
    filepath = EXPERIMENTS+"debug/"+filepath if debug else EXPERIMENTS+filepath

    return filename, filepath


def _get_attack_savedir(model_name, dataset_name, epochs, debug, attack_method, attack_library,
                        size_proj=None, projection_mode=None, centroid_translation=False, robust=False):

    filepath = model_name+"_"+dataset_name+"_ep="+str(epochs)

    if size_proj:
        filepath += "_size="+str(size_proj)+"_"+projection_mode

    if centroid_translation:
        filepath += "_centroid"

    if robust:
        filepath += attack_library+"_"+attack_method+"_robust"

    filename = attack_library+"_"+attack_method+"_attack"
    filepath = EXPERIMENTS+"debug/"+filepath if debug else EXPERIMENTS+filepath

    return filename, filepath