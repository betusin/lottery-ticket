import common as common

if __name__ == "__main__":
    mnist_data = common.load_dataset('mnist', 60)
    arch = 'lenet'

    dict_trial_id = {
        "0": "1667011973",
        "1": "1667059500",
        "2": "1667108043",
        "3": "1667156427",
        "4": "1667202408"
    }

    # write header
    common.write_test_acc_to_file(f"{arch}-test_acc.csv", "identifier", "trial", "round", "iteration", "test_acc", "test_loss")

    for trial, identifier in dict_trial_id.items():
        common.write_test_accuracy_to_csv(arch, mnist_data, identifier, trial)
