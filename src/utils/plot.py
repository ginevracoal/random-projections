
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


def covariance_eigendec(np_matrix):
    print("\nmatrix[rows=vars, cols=obs] = \n", np_matrix)
    C = np.cov(np_matrix)
    print("\ncovariance matrix = \n", C)

    eVe, eVa = np.linalg.eig(C)

    plt.scatter(np_matrix[:, 0], np_matrix[:, 1])
    for e, v in zip(eVe, eVa.T):
        plt.plot([0, 3 * np.sqrt(e) * v[0]], [0, 3 * np.sqrt(e) * v[1]], 'k-', lw=2)
    plt.title('Transformed Data')
    plt.axis('equal')
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS+"covariance.png")


def rgb2gray(rgb):
    """ convert rgb image to greyscale image """
    if rgb.shape[2] == 1:
        return rgb
    else:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def compute_angle(v1, v2):
    """ Compute the angle between two numpy arrays, eventually flattening them if multidimensional. """
    if len(v1) != len(v2): raise ValueError("\nYou cannot compute the angle between vectors with different dimensions.")
    v1 = v1.flatten()
    v2 = v2.flatten()
    return math.acos(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) )


def compute_covariance_matrices(x,y):
    """
    Compute within-class and between-class variances on the given data.
    :param x: input data, type=np.ndarray, shape=(n_samples, n_features)
    :param y: data labels, type=np.ndarray, shape=(n_samples, n_classes)
    :return: average within-class variance, average between-class variance
    """
    standardize = lambda x: (x - np.mean(x))/ np.std(x)
    normalize = lambda x: (x - np.min(x))/ (np.max(x)-np.min(x))

    # reshape and standardize data
    n_features =  x.shape[1]
    x = x.reshape(len(x),n_features)
    x = standardize(x)

    # compute mean and class mean
    mu = np.mean(x, axis=0).reshape(n_features,1)
    y_true = np.argmax(y, axis=1)
    mu_classes = []
    for i in range(10):
        mu_classes.append(np.mean(x[np.where(y_true == i)], axis=0))
    mu_classes = np.array(mu_classes).T

    # compute scatter matrices
    data_SW = []
    Nc = []
    for i in range(10):
        a = np.array(x[np.where(y_true == i)] - mu_classes[:, i].reshape(1, n_features))
        data_SW.append(np.dot(a.T, a))
        Nc.append(np.sum(y_true == i))
    SW = np.sum(data_SW, axis=0)
    SB = np.dot(Nc * np.array(mu_classes - mu), np.array(mu_classes - mu).T)

    SW = normalize(SW)
    SB = normalize(SB)
    print("\nWithin-class avg normalized variance:", np.mean(SW))
    print("Between-class avg normalized variance:", np.mean(SB))

    return SW, SB


def compute_distances(x1,x2,ord):
    """
    Computes min, avg and max distances between the inputs and their perturbations
    :param x1: input points, shape=(n_samples, rows, cols, channels), type=np.ndarray
    :param x2: perturbations, shape=(n_samples, rows, cols, channels), type=np.ndarray
    :param ord: norm order for np.linalg.norm
    :return: min, average, max distances between all couples of points, type=dict
    """
    if x1.shape != x2.shape:
        raise ValueError("\nThe arrays need to have the same shape.")
    flat_x1 = x1.reshape(x1.shape[0], np.prod(x1.shape[1:]))
    flat_x2 = x2.reshape(x2.shape[0], np.prod(x2.shape[1:]))
    min = np.min([np.linalg.norm(flat_x1[idx] - flat_x2[idx], ord=ord) for idx in range(len(x1))])
    mean = np.mean([np.linalg.norm(flat_x1[idx] - flat_x2[idx], ord=ord) for idx in range(len(x1))])
    max = np.max([np.linalg.norm(flat_x1[idx] - flat_x2[idx], ord=ord) for idx in range(len(x1))])
    return {"min":min,"mean": mean,"max": max}

