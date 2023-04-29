import common as common

if __name__ == "__main__":
    mnist = common.load_dataset('mnist', 60)
    common.create_checkpoint_dir('lenet')
    for i in range(5):
        print("\nIdentifying winning ticket at i ", i)
        common.identify_winning_ticket('lenet', mnist, 28, i)

    for i in range(10):
        print("\nGetting random ticket at i ", i)
        common.get_random_ticket('lenet', mnist, 28, i)
