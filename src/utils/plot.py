
def plot_heatmap(columns, path, filename, xlab=None, ylab=None, title=None, yticks=None):
    columns = np.array(columns)
    # print(columns.shape)
    fig, ax = plt.subplots(figsize=(15, 6), dpi=400)
    sns.heatmap(columns, ax=ax)
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    if yticks:
        ax.set_yticklabels(yticks)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path+filename)


def plot_loss_accuracy(dict, path):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
    ax1.plot(dict['loss'])
    ax1.set_title("loss")
    ax2.plot(dict['accuracy'])
    ax2.set_title("accuracy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)


def violin_plot(data, path, filename, xlab=None, ylab=None, title=None, yticks=None):
    fig, axes = plt.subplots(figsize=(15, 6))
    # sns.set(style="whitegrid")
    sns.violinplot(data=data, ax=axes, orient='v')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + filename)

def rgb2gray(rgb):
    """ convert rgb image to greyscale image """
    if rgb.shape[2] == 1:
        return rgb
    else:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def plot_attacks(dataset_name, epochs, debug, attacks, attack_library):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           debug=debug)
    model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                            dataset_name=dataset_name, epochs=epochs)
    model.load_classifier(relative_path=RESULTS, filename=model.filename+"_seed="+str(seed))
    images = []
    labels = []
    for attack in attacks:
        eps = model._get_attack_eps(dataset_name=model.dataset_name, attack=attack)
        x_test_adv = model.generate_adversaries(x=x_test, y=y_test, attack=attack, eps=eps)
        images.append(x_test_adv)
        avg_dist = compute_distances(x_test, x_test_adv, ord=model._get_norm(attack))['mean']
        labels.append(str(attack) + " avg_dist=" + str(avg_dist))

    plot_images(image_data_list=images,labels=labels)