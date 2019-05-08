import torch as t
from torch.autograd import Variable as V
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from torch import optim

class RegressionByTorch(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(RegressionByTorch, self).__init__()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

    @staticmethod
    def get_data(verbose=False):
        t.manual_seed(100)
        x = t.unsqueeze(t.linspace(-1, 1, 100), dim=1)
        y = x ** 2 + 0.2 * (t.randn_like(x))
        # x, y = V(x), V(y)
        if verbose:
            print("x is: ", x, '\n', "y is: ", y)
            plt.scatter(x, y)
            plt.show()
        return x, y


if __name__ == '__main__':
    R = RegressionByTorch(1,10,1)
    x, y = R.get_data()
    print(R)
    optimizer = optim.SGD(R.parameters(), lr=0.2)
    loss_func = nn.MSELoss()

    plt.ion()
    for t in range(200):
        prediction = R(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t%5 == 0:
            plt.cla()
            plt.scatter(x, y)
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.pause(.1)
    plt.ioff()
    plt.show()
