import torch
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

x_data = torch.Tensor([[1.0],[2.0],[3.0],[4.0]])
y_data = torch.Tensor([[1.0],[1.0],[0.0],[0.0]])

model = MyModel()
criterion = torch.nn.BCELoss(reduction='mean') # >> criterion = torch.nn.BCELoss(size_average = 'True)
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

for epoch in range(1000):

    y_pred = model(x_data)

    loss = criterion(y_pred,y_data)

    print("epoch:", epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var = float(input('input: '))
v = torch.Tensor([hour_var])
print(hour_var, " hour: ", model(v).item() > 0.5)