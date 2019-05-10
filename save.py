import torch as t
from torch import nn
from torch import optim


t.manual_seed(1)
x = t.unsqueeze(t.linspace(-5,5,100), dim=1)
y = x**2 + .1*t.rand_like(x)


net = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)


loss_func = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=.02)
# t.save(net, 'net.pkl')
# t.save(net.state_dict(), 'net_param.pkl')


for i in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(prediction)





