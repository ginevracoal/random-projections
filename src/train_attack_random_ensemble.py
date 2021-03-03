import argparse
from utils.exec_settings import *
from models.random_ensemble import RandomEnsemble
from models.baseline_convnet import BaselineConvnet

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="mnist", type=str, help=str(DATASETS))
parser.add_argument("--projection_mode", default="channels", type=str, help=str(PROJ_MODE))
parser.add_argument("--attack_method", default="fgsm", type=str, help=str(ATTACKS))
parser.add_argument("--epochs", default=20, type=int, help="Training epochs.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cpu', type=str, help="cpu")  # todo: gpu
parser.add_argument("--attack_library", default='art', type=str, help="art, cleverhans")  
args = parser.parse_args()

n_proj_list = [6, 9, 12, 15]
size_proj_list = [8, 12, 16, 20]

if args.attack_method:
    attacks_list = [args.attack_method]
else:
    attacks_list = ["fgsm", "pgd", "deepfool", "carlini", "newtonfool", "virtual"]

n_jobs=10 if args.device=="cpu" else 1
set_session(device=args.device, n_jobs=n_jobs)

def attack_randens(dataset_name, epochs, n_proj, size_proj, projection_mode, attacks_list, attack_library, 
                    device, debug, load):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, debug)

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=n_proj, size_proj=size_proj, projection_mode=projection_mode,
                           data_format=data_format, dataset_name=dataset_name, epochs=epochs,
                           centroid_translation=False)

    if load:
        model.load_classifier(debug)

    else:
        model.train(x_train, y_train, device=device)
        model.save_classifier(debug)

    for attack_method in attacks_list:
        print(f"=== \n{attack_method} attack ===")

        x_test_adv = model.load_adversaries(attack_method=attack_method, attack_library=attack_library, debug=debug)
        model.evaluate(x=x_test_adv, y=y_test)

for n_proj in n_proj_list:
    for size_proj in size_proj_list:
        K.clear_session()
        attack_randens(dataset_name=args.dataset_name, epochs=args.epochs, n_proj=n_proj, size_proj=size_proj,
                       projection_mode=args.projection_mode, attacks_list=attacks_list,
                       attack_library=args.attack_library, device=args.device, debug=args.debug, load=args.load)

