import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import optim

t.manual_seed(100)

n_data = t.ones(100, 2)
x0 = t.normal(2*n_data, 1)
y0 = t.zeros(100)
x1 = t.normal(-2*n_data, 1)
y1 = t.ones(100)
x = t.cat((x0, x1), 0)
y = t.cat((y0, y1), 0).type(t.LongTensor)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

class Classification(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Classification, self).__init__()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

classfication = Classification(2, 10, 2)
print(classfication)

optimizer = optim.SGD(classfication.parameters(), lr=0.002)
loss_func = nn.CrossEntropyLoss()

plt.ion()
for i in range(100):
    out = classfication(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(max(out.data, 1)[1])


    if i%2 == 0:
        plt.cla()
        prediction = t.max(out.data, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        plt.pause(.1)
    plt.ioff()
    plt.show()