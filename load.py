import torch as t
from torch import nn
from torch import optim


net1 = t.load('net.pkl')
t.manual_seed(1)
x = t.unsqueeze(t.linspace(-5,5,100), dim=1)
y = x**2 + .1*t.rand_like(x)


loss_func = nn.MSELoss()
optimizer = optim.SGD(net1.parameters(), lr=.02)


for i in range(100):
    prediction = net1(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(prediction)




net2 = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
net2.load_state_dict(t.load('net_param.pkl'))


loss_func = nn.MSELoss()
optimizer = optim.SGD(net2.parameters(), lr=.02)



for i in range(100):
    prediction = net2(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(prediction)
