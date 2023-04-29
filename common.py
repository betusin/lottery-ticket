import torch
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l
import time
import torch.nn.utils.prune as prune
import os

device = d2l.try_gpu()

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


def load_dataset(name, batch_size, **kwargs):
    if name == 'mnist':
        return load_mnist(batch_size)
    elif name == 'cifar10':
        return load_cifar10(batch_size)
    else:
        raise Exception(f"Unknown Dataset: {name}")


def load_mnist(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.0,), (1.0,)),
        lambda img: img.flatten()
    ])

    train_data = torchvision.datasets.MNIST(
        'data/', download=True, transform=transform, train=True)
    test_data = torchvision.datasets.MNIST(
        'data/', download=True, transform=transform, train=False)

    train_data, val_data = torch.utils.data.random_split(
        train_data, [len(train_data)-5000, 5000])

    data_loaders = {'train_data': torch.utils.data.DataLoader(train_data, batch_size, shuffle=True),
                    'test_data': torch.utils.data.DataLoader(test_data, batch_size, shuffle=True),
                    'validation_data': torch.utils.data.DataLoader(val_data, batch_size, shuffle=True)}

    return data_loaders


def load_cifar10(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.0,), (1.0,))
    ])

    train_data = torchvision.datasets.CIFAR10(
        'data/', download=True, transform=transform, train=True)
    test_data = torchvision.datasets.CIFAR10(
        'data/', download=True, transform=transform, train=False)

    # Split data in test/validation

    train_data, val_data = torch.utils.data.random_split(
        train_data, [len(train_data)-5000, 5000])

    data_loaders = {'train_data': torch.utils.data.DataLoader(train_data, batch_size, shuffle=True),
                    'test_data': torch.utils.data.DataLoader(test_data, batch_size, shuffle=True),
                    'validation_data': torch.utils.data.DataLoader(val_data, batch_size, shuffle=True)}

    return data_loaders

# -----------------------------------------------------------------------------
# Network architectures
# -----------------------------------------------------------------------------


class Lenet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.net(x)


class Conv2(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=kernel_size, stride=2),

            torch.nn.Flatten(),

            torch.nn.Linear(6400, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


class Conv4(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=kernel_size, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=kernel_size, stride=2),

            torch.nn.Flatten(),
            torch.nn.Linear(1152, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


class Conv6(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=kernel_size, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=kernel_size, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=kernel_size, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=kernel_size, stride=2),

            torch.nn.Flatten(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


class Resnet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.resnet18()

    def forward(self, x):
        return self.net(x)


class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.vgg19()

    def forward(self, x):
        return self.net(x)


def create_network(arch, **kwargs):
    if arch == 'lenet':
        return Lenet()
    elif arch == 'conv2':
        return Conv2()
    elif arch == 'conv4':
        return Conv4()
    elif arch == 'conv6':
        return Conv6()
    elif arch == 'resnet18':
        return Resnet18()
    elif arch == 'vgg19':
        return VGG19()
    else:
        raise Exception(f"Unknown architecture: {arch}")


def get_hyperparams(arch, net):
    """
    Return hyperparameters given a specific architecture, as defined in The Lottery Ticket Hypothesis
    Returns iterations, learning rate, optimizer, linear prune percentage, convolutional prune percentage
    """
    if arch == "lenet":
        lr = 0.0012
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1.2e-3)
        return 50000, lr, optimizer, 0.2, 0
    if arch == "conv2":
        lr = 0.0004
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=2e-4)
        return 20000, lr, optimizer, 0.2, 0.1
    if arch == "conv4":
        lr = 0.0004
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=3e-4)
        return 25000, lr, optimizer, 0.2, 0.1
    if arch == "conv6":
        lr = 0.0004
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=3e-4)
        return 30000, lr, optimizer, 0.2, 0.15
    if arch == "resnet18":
        lr = 0.1
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        return 30000, lr, optimizer, 0, 0.2
    if arch == "conv6":
        lr = 0.1
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        return 112000, lr, optimizer, 0, 0.2

# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------

# TODO: Define training, testing and model loading here


def evaluate_loss(net, data_iter, loss,  device=device):
    """
    Evaluate the loss of a model on the given dataset.
    Originally from d2l book, but without the reshape
    """
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        out = net(X)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def evaluate(net, data, loss):
    """
    net: a trained network
    data: a DataLoader, either test or validation
    loss: a callable loss function
    """
    net.eval()
    accuracy = d2l.evaluate_accuracy_gpu(net, data)
    loss = evaluate_loss(net, data, loss)
    return accuracy, loss


def save_model(model, arch, filename=""):
    if filename == "":  # Set default filename
        time = int(time.time())
        filename = f"model-{time}.pth"
    torch.save(model, f"checkpoints/{arch}/" + filename)


def train(net, arch, data, num_iter, lr, optimizer, round, cur_time, device=device):
    """Train a model with a GPU (defined in Chapter 6). Returns a unique identifier corresponding to this training session"""

    train_iter = data['train_data']

    print('training on', device)
    net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)

    save_model(net, arch, f"{cur_time}-trial_{round}-init.pth")

    epochs = 0
    iterations = 0
    best_val_loss = 10  # Some arbitrary high value
    best_iteration = -1

    timer.start()

    while iterations < num_iter:

        metric = d2l.Accumulator(3)

        for X, y in train_iter:
            if iterations >= num_iter:
                break

            net.train()

            net.train()

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)

            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            if iterations % 100 == 0:
                _, cur_val_loss = evaluate(net, data['validation_data'], loss)
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_iteration = iterations

                    save_model(net, arch, f"{cur_time}-trial_{round}-best.pth")
            iterations += 1

            if iterations % 1000 == 0:
                save_model(
                    net, arch, f"{cur_time}-trial_{round}-{iterations}.pth")

        epochs += 1

    timer.stop()

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')
    print(f'{iterations / timer.sum():.1f} iterations/sec '
          f'on {str(device)}')
    print(
        f'hyperparameters {epochs} epochs, {iterations} interations & lr={lr}')

    save_model(net, arch, f"{cur_time}-trial_{round}-final.pth")

    return best_val_loss, best_iteration


def train_with_plot(net, arch, data, num_iter, lr, optimizer, round, cur_time, device=device):
    """Train a model with a GPU (defined in Chapter 6). Returns a unique identifier corresponding to this training session"""

    train_iter = data['train_data']

    print('training with plot on', device)
    net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel=f'iterations, lr={lr:.4f}', xlim=[1, num_iter],
                            legend=['train loss', 'train acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    save_model(net, arch, f"{cur_time}-{round}-init.pth")

    epochs = 0
    iterations = 0
    best_val_loss = 10  # Some arbitrary high value
    best_iteration = -1

    timer.start()

    while iterations < num_iter:
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            if iterations >= num_iter:
                break

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)

            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            if iterations % 100 == 0:
                animator.add(iterations + 1, (train_l, train_acc))
                _, cur_val_loss = evaluate(net, data['validation_data'], loss)
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_iteration = iterations
                    save_model(net, arch, f"{cur_time}-{round}-best.pth")

            iterations += 1

        epochs += 1
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')
    print(f'{iterations / timer.sum():.1f} iterations/sec '
          f'on {str(device)}')
    print(
        f'hyperparameters {epochs} epochs, {iterations} interations & lr={lr}')

    save_model(net, arch, f"{cur_time}-{round}-final.pth")

    return best_val_loss, best_iteration


def load_and_evaluate(arch, filename, dataloaders, loss):
    net = torch.load(f"checkpoints/{arch}/{filename}")
    _, v_loss = evaluate(net, dataloaders['validation_data'], loss)
    t_acc, _ = evaluate(net, dataloaders['test_data'], loss)
    return v_loss, t_acc

# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------


def prune_network(net, amount_linear, amount_conv):

    num_of_layers = len(list(net.net.children()))

    for i, layer in enumerate(net.net.children()):
        if isinstance(layer, torch.nn.Linear):
            if i is num_of_layers-1:  # This is the final layer, so half pruning amount
                prune.l1_unstructured(
                    layer, name='weight', amount=amount_linear/2)
            else:
                prune.l1_unstructured(
                    layer, name='weight', amount=amount_linear)
        if isinstance(layer, torch.nn.Conv2d):
            prune.l1_unstructured(layer, name='weight', amount=amount_conv)


def copy_pruning_mask(source, dest):
    """
    Applies the pruning mask from a pruned network to another network of the same architecture
    source: a pruned network, with pruning masks
    dest: a network to be pruned
    """

    state_dict = source.state_dict()

    for i, layer in enumerate(dest.net.children()):
        if (isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d)):
            key = f"net.{i}.weight_mask"
            mask = state_dict[key]
            prune.custom_from_mask(layer, 'weight', mask)


def prune_network_random(net, amount_linear, amount_conv):

    num_of_layers = len(list(net.net.children()))

    for i, layer in enumerate(net.net.children()):
        if isinstance(layer, torch.nn.Linear):
            if i is num_of_layers-1:  # This is the final layer, so half pruning amount
                prune.random_unstructured(
                    layer, name='weight', amount=amount_linear/2)
            else:
                prune.random_unstructured(
                    layer, name='weight', amount=amount_linear)
        if isinstance(layer, torch.nn.Conv2d):
            prune.random_unstructured(layer, name='weight', amount=amount_conv)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Supplementary functions for the experiments

# Creates working dir for saving checkpoints of particular *arch*


def create_checkpoint_dir(arch):
    cwd = os.getcwd()
    work_dir = f"checkpoints/{arch}"
    path = os.path.join(cwd, work_dir)

    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print("Directory '%s' can not be created." % path)
        print("Error: %s" % error)


def identify_winning_ticket(arch, data, num_rounds, trial):
    """
    Identifies winning tickets as described in THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS
    1. Randomly initialize a neural network f(x; θ0) (where θ0 equiv Dθ).
    2. Train the network for j iterations, arriving at parameters θj .
    3. Prune p% of the parameters in θj , creating a mask m.
    4. Reset the remaining parameters to their values in θ0, creating the winning ticket f(x; m * θ0).
    """
    net = create_network(arch)  # Step 1

    # Used for giving unique name to each training session
    cur_time = int(time.time())
    loss = torch.nn.CrossEntropyLoss()
    total_weights = count_nonzero_weights(net)

    for round in range(num_rounds):

        print("-" * 50)
        print(f"Starting training round {round}")
        iterations, lr, optimizer, prune_linear, prune_conv = get_hyperparams(
            arch, net)  # Step 2
        best_val_loss, best_iteration = train(
            net, arch, data, iterations, lr, optimizer, f"trial_{trial}-{round}", cur_time)  # Step 2
        print(f"Starting validation round {round}")

        best_net = torch.load(
            f"checkpoints/{arch}/{cur_time}-trial_{round}-best.pth")

        t_acc, t_loss = evaluate(best_net, data['test_data'], loss)

        cur_weights = count_nonzero_weights(net)
        percentage_weights = f"{(cur_weights/total_weights * 100):.1f}"

        write_results_to_file(
            f"{arch}-results-1.csv", f"{arch},{cur_time},{trial},{round},{iterations},{best_val_loss},{best_iteration},{t_acc},{t_loss},{cur_weights},{percentage_weights}")

        prune_network(net, prune_linear, prune_conv)  # Step 3
        orig_net = torch.load(
            f"checkpoints/{arch}/{cur_time}-trial_0-init.pth")  # Step 4
        copy_pruning_mask(net, orig_net)  # Step 4
        net = orig_net

    return orig_net


def get_random_ticket(arch, data, num_rounds, trial):
    """
    Identifies winning tickets as described in THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS
    1. Randomly initialize a neural network f(x; θ0) (where θ0 equiv Dθ).
    2. Train the network for j iterations, arriving at parameters θj .
    3. Prune p% of the parameters in θj , creating a mask m.
    4. Reset the remaining parameters to their values in θ0, creating the winning ticket f(x; m * θ0).
    """
    net = create_network(arch)  # Step 1

    # Used for giving unique name to each training session
    cur_time = int(time.time())
    loss = torch.nn.CrossEntropyLoss()
    total_weights = count_nonzero_weights(net)

    for round in range(num_rounds):

        print("-" * 50)
        print(f"Starting training round {round}", "*"*10)
        iterations, lr, optimizer, prune_linear, prune_conv = get_hyperparams(
            arch, net)  # Step 2
        best_val_loss, best_iteration = train(
            net, arch, data, iterations, lr, optimizer, f"trial_{trial}-{round}", cur_time)  # Step 2
        print(f"Starting validation round {round}")

        best_net = torch.load(
            f"checkpoints/{arch}/{cur_time}-trial_{round}-best.pth")

        t_acc, t_loss = evaluate(best_net, data['test_data'], loss)

        cur_weights = count_nonzero_weights(net)
        percentage_weights = f"{(cur_weights/total_weights * 100):.1f}"

        write_results_to_file(f"{arch}-results-random-1.csv",
                              f"{arch},{cur_time},{trial},{round},{iterations},{best_val_loss},{best_iteration},{t_acc},{t_loss},{cur_weights},{percentage_weights}")

        print(f"Starting pruning round {round}")

        prune_network_random(net, prune_linear, prune_conv)  # Step 3

        orig_net = torch.load(
            f"checkpoints/{arch}/{cur_time}-trial_0-init.pth")  # Step 4

        copy_pruning_mask(net, orig_net)  # Step 4
        net = orig_net

    return orig_net


def count_nonzero_weights(net):
    total_weights = 0
    for layer in net.net.children():
        if (isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d)):
            if (isinstance(layer.weight, torch.nn.parameter.UninitializedParameter)):
                print("Uninitialized params weight on layer", layer)
                print("Initialize weights before!")
                return
            else:
                total_weights += torch.count_nonzero(layer.weight)
    return total_weights.item()

# -----------------------------------------------------------------------------
# Extracting data
# -----------------------------------------------------------------------------


def write_results_to_file(filename, data):
    with open(filename, "a") as results:
        results.write(f"{data}\n")


def write_test_accuracy_to_csv(arch, data, identifier, trial):
    """
        Extracts test accuracy from model of given architecture
        at particular trial and identifier.

        Chosen rounds with remaining weights percentage:
            100% - 0
            51.3% - 3
            21.1% - 7
            7.0% - 12
            3.6% - 15
            1.9% - 18
    """

    for round in [0, 3, 7, 12, 15, 18]:
        get_test_accuracy_at_round(arch, data, identifier, trial, round)


def get_test_accuracy_at_round(arch, data, identifier, trial, round):
    loss = torch.nn.CrossEntropyLoss()

    for iteration in range(0, 20000, 1000):
        net = torch.load(
            f"checkpoints/{arch}/{identifier}-trial_{trial}-{round}-{iteration}.pth")
        test_acc, test_loss = evaluate(net, data['test_data'], loss)

        write_results_to_file(
            f"{arch}-test_acc.csv", f"{identifier},{trial},{round},{iteration},{test_acc},{test_loss}")
