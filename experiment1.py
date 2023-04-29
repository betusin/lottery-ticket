import common as common
import sys

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("Argument for net architecture and dataset not passed!")
        arch = 'lenet'
        dataset_name = 'mnist'
    else:
        arch = sys.argv[1]
        dataset_name = sys.argv[2]

    print("Running experiment with architecture '%s'" % arch)
    print("Running experiment with dataset '%s'" % dataset_name)

    common.create_checkpoint_dir(arch)
    dataset = common.load_dataset(dataset_name, 60)

    for i in range(5):
        common.identify_winning_ticket(arch, dataset, 28, i)

    for i in range(10):
        common.get_random_ticket(arch, dataset, 28, i)
