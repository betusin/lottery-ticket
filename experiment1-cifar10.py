import common as common

if __name__ == "__main__":
	cifar10 = common.load_dataset('cifar10', 60)

	for arch in ['conv2', 'conv4', 'conv6']:
		common.create_checkpoint_dir(arch)

		for i in range(5):
			print("\nIdentifying winning ticket at i ", i)
			common.identify_winning_ticket(arch, cifar10, 28, i)

		for i in range(10):
			print("\nGetting random ticket at i ", i)
			common.get_random_ticket(arch, cifar10, 28, i)

