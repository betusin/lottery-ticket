import torch
import common as common

# TODO: your code here.

mnist = common.load_dataset('mnist', 60)

lr = 0.001
net = common.Lenet()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

identifier = common.train(net, "testrun", mnist, 10000, lr, optimizer)
print(f'Unique identifier for this training run: = {identifier}')
