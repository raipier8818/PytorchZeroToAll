# import torch
# import pdb
# from torch.autograd import Variable
#
# x_data = [1.0,2.0,3.0]
# y_data = [2.0,4.0,6.0]
#
# w = Variable(torch.Tensor([1,0]), requires_grad = True)
#
#
#
# def forward(x):
#     return x*w
#
# def loss(x,y):
#     y_pred = forward(x)
#     return (y_pred - y)**2
#
#
# print("Predict (before training)", 4, forward(4).data[0])
#
# for epoch in range(10):
#     for x_val, y_val in zip(x_data,y_data):
#         l = loss(x_val,y_val)
#         l.backward()
#         print("\tgrad: ",x_val,y_val,w.grad.data[0])
#         w_data = w.data - 0.01*w.grad.data
#
#         w.grad.data.zero_()
#
#     print("Progress:", epoch, l.data[0])
#
# print("Predict (after training)", 4, forward(4).data[0])

import torch
import pdb

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True)

# our model forward pass
def forward(x):
    return x * w

# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2

# Before training
print("Prediction (before training)",  4, forward(4).item())

# Training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation to update weights
        print("\tgrad: ", x_val, y_val, w.grad.item())
        w.data = w.data - 0.01 * w.grad.item()

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}" '// w:', w.data)

# After training
print("Prediction (after training)",  4, forward(4).item())