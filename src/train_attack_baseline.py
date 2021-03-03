import argparse
from utils.exec_settings import *
from models.baseline_convnet import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="mnist", type=str, help=str(DATASETS))
parser.add_argument("--attack_method", default=None, type=str, help=str(ATTACKS))
parser.add_argument("--epochs", default=20, type=int, help="Training epochs.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cpu', type=str, help="cpu")  # todo: gpu
parser.add_argument("--attack_library", default='art', type=str, help="art, cleverhans")  
args = parser.parse_args()

if args.attack_method:
    attacks_list = [args.attack_method]
else:
    attacks_list = ["fgsm", "pgd"]#, "deepfool", "carlini", "newtonfool", "virtual"]

n_jobs=10 if args.device=="cpu" else 1
set_session(device=args.device, n_jobs=n_jobs)

def train_eval(dataset_name, epochs, attacks_list, attack_library, debug, device, load=False):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           debug=debug)

    baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                               dataset_name=dataset_name, epochs=epochs)
    
    if load:
        baseline.load_classifier(debug)

    else:
        baseline.train(x_train, y_train, device=device)
        baseline.save_classifier(debug)

    baseline.evaluate(x=x_test, y=y_test)

    for attack_method in attacks_list:
        x_test_adv = baseline.generate_adversaries(x=x_test, y=y_test, device=device,
                                                   attack_method=attack_method, attack_library=attack_library)

        baseline.save_adversaries(data=x_test_adv, attack_method=attack_method, attack_library=attack_library, debug=debug)
        # x_test_adv = baseline.load_adversaries(attack=attack, relative_path=DATA_PATH, seed=0)
        # softmax_difference(classifier=baseline, x1=x_test, x2=x_test_adv)
        # softmax_robustness(classifier=baseline, x1=x_test, x2=x_test_adv)
        baseline.evaluate(x=x_test_adv, y=y_test)

def adversarial_train_eval(dataset_name, epochs, attacks_list, attack_library, debug, device, load=False):
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           debug=debug)

    print("\n === Baseline === ")

    baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                               dataset_name=dataset_name, epochs=epochs)
    baseline.load_classifier(debug)
    baseline.evaluate(x=x_test, y=y_test)

    print("\n === Adversarially trained models === ")

    robust_baselines = []
    for attack_method in attacks_list:
        print("\n", attack_method, "robust baseline")

        robust_baseline = baseline.adversarial_train(x_train=x_train, y_train=y_train, device=device, 
                                                    attack_method=attack_method, attack_library=attack_library)
        robust_baseline.save_classifier(debug)

        robust_baseline.evaluate(x=x_test, y=y_test)
        robust_baselines.append(robust_baseline)

    print("\n === Adversarial evaluations === ")

    for attack_method in attacks_list:
        x_test_adv = baseline.load_adversaries(attack_method=attack_method, attack_library=attack_library, debug=debug)
        baseline.evaluate(x=x_test_adv, y=y_test)
        # softmax_difference(classifier=baseline, x1=x_test, x2=x_test_adv)
        for idx, tmp_attack_method in enumerate(attacks_list):
            print(f"\n{attack_method} attack against {tmp_attack_method} robust baseline")
            robust_baselines[idx].evaluate(x=x_test_adv, y=y_test)

# def robust_eval(dataset_name, test, seed, device, robust_models, attacks, library):
#     x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
#                                                                                            test=test)
#     baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
#                                dataset_name=dataset_name, test=test, epochs=None, library=library)
#     baseline.load_classifier(relative_path=TRAINED_MODELS)
#     baseline.evaluate(x=x_test, y=y_test)
#     for method in robust_models:
#         print("\n===",method,"robust eval ===")
#         robust_baseline = baseline.load_robust_classifier(relative_path=TRAINED_MODELS, attack=method, seed=seed)
#         robust_baseline.evaluate(x=x_test, y=y_test)
#         for attack in attacks:
#             for seed in [0,1,2]:
#                 print("\nseed =", seed)
#                 x_test_adv = baseline.load_adversaries(attack=attack, relative_path=DATA_PATH, seed=seed)
#                 robust_baseline.evaluate(x=x_test_adv, y=y_test)


train_eval(dataset_name=args.dataset_name, epochs=args.epochs, attacks_list=attacks_list, attack_library=args.attack_library, 
           debug=args.debug, device=args.device, load=args.load)

adversarial_train_eval(dataset_name=args.dataset_name, epochs=args.epochs, attacks_list=attacks_list, attack_library=args.attack_library, 
                        debug=args.debug, device=args.device, load=args.load)