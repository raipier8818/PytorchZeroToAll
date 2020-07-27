# import torch
# from torch.autograd import Variable
# from torch import nn
#
# x_data = Variable(torch.Tensor([[1,0],[2,0],[3,0]]))
# y_data = Variable(torch.Tensor([[2,0],[4,0],[6,0]]))
#
#
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear = torch.nn.Linear(1,1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
# model = Model()
#
# criterion = torch.nn.MSELoss(size_average = False)
# optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
#
# for epoch in range(500):
#     y_pred = model(x_data)
#
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.data[0])
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# hour_var = Variable(torch.Tensor([4.0]))
# print("predict (after training)", 4, model.forward(hour_var).data[0][0])

from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(501):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('---------------------------')
# After training
a = float(input("input number: "))
hour_var = tensor([[a]])
y_pred = model(hour_var)
print("Prediction (after training)",  a, model(hour_var).data[0][0].item())