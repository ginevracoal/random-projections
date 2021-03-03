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

def attack_randens(dataset_name, n_proj, size_proj, projection_mode, attacks_list, device, debug):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, debug)

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=n_proj, size_proj=size_proj, projection_mode=projection_mode,
                           data_format=data_format, dataset_name=dataset_name, epochs=args.epochs,
                           centroid_translation=False)

    if args.load:
        model.load_classifier(args.debug)

    else:
        model.train(x_train, y_train, device=args.device)
        model.save_classifier(args.debug)

    baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                               dataset_name=dataset_name, epochs=args.epochs)
    baseline.load_classifier(args.debug)

    model.evaluate(x=x_test, y=y_test)

    exit()

    for attack in attacks_list:

        if args.load:
            x_test_adv = model.load_adversaries(relative_path=RESULTS, attack=attack)
        else:
            x_test_adv = baseline.generate_adversaries(x=x_test, y=y_test, attack=attack, seed=seed)
            baseline.save_adversaries(data=x_test_adv, attack=attack, seed=seed)

        model.evaluate(x=x_test_adv, y=y_test)
        softmax_difference(classifier=model, x1=x_test, x2=x_test_adv)

    # x_test_adv = model.load_adversaries(dataset_name=dataset_name,attack=attack, eps=eps, test=test)
    # print("Distance from perturbations: ", compute_distances(x_test, x_test_adv, ord=model._get_norm(attack)))
    # model.evaluate(classifier=classifier, x=x_test_adv, y=y_test)

    # === generate perturbations === #
    # compute_variances(x_test, y_test)
    # projections, inverse_projections = model.compute_projections(input_data=x_test)
    #
    # projections, inverse_projections = model.compute_projections(input_data=x_test)
    # print(projections[0,0,0,0], inverse_projections[0,0,0,0])
    # # exit()
    # perturbations, augmented_inputs = compute_perturbations(input_data=x_test, inverse_projections=inverse_projections)

    # # print(np.mean([compute_angle(x_test[i],augmented_inputs[i]) for i in range(len(x_test))]))
    # # exit()
    # eig_vals, eig_vecs = compute_linear_discriminants(x_test, y_test)
    # print(eig_vals,"\n",eig_vecs)
    # exit()
    # # print(x_test[0,0,0,:],augmented_inputs[0,0,0,:],"\n")
    # avg_distance = lambda x: np.mean([np.linalg.norm(x[0][idx]-x[1][idx]) for idx in range(len(x_test))])
    # print("Average distance from attack: ", avg_distance([x_test, augmented_inputs]))

    # # # === evaluate baseline on perturbations === #
    # baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
    #                         dataset_name=dataset_name, test=True)
    # rel_path = "../trained_models/baseline/" + str(dataset_name) + "_baseline.h5"
    # baseline_classifier = baseline.load_classifier(relative_path=rel_path)
    # baseline.evaluate(baseline_classifier, augmented_inputs, y_test)

    # === plot perturbations === #
    # plot_images(image_data_list=[x_test, projections[0], inverse_projections[0]])
    # plot_images(image_data_list=[x_test,perturbations,augmented_inputs])

    # adversaries=[]
    # for method in ['fgsm', 'pgd', 'deepfool', 'carlini']:
    #     adversaries.append(model.load_adversaries(attack=method, eps=0.5))
    # plot_images(image_data_list=adversaries)

for n_proj in n_proj_list:
    for size_proj in size_proj_list:
        K.clear_session()
        attack_randens(dataset_name=args.dataset_name, n_proj=n_proj, size_proj=size_proj,
                       projection_mode=args.projection_mode, attacks_list=attacks_list, 
                       device=args.device, debug=args.debug)

