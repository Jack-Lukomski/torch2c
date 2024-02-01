import torch

TRAINING_SIZE = 4

xi = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
xo = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class xor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2, 2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
'''
EPOCS = 10000

model = xor()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.modules.loss.MSELoss()

def test_model():
    error = 0
    with torch.no_grad():
        for i in range(TRAINING_SIZE):
            output = model(xi[i])
            error += loss_fn(output, xo[i])

        error /= TRAINING_SIZE
    return error

for epoc in range(EPOCS):
    for i in range(TRAINING_SIZE):
        optimizer.zero_grad()
        output = model(xi[i])
        loss = loss_fn(output, xo[i])
        loss.backward()
        optimizer.step()
    print(f'EPOC {epoc} Error: {test_model()}')


with torch.no_grad():
    for i in range(TRAINING_SIZE):
        output = model(xi[i])
        print(f'{xi[i]} xored prediction = {output}, actual = {xo[i]}')

torch.save(model, 'xor.pth')
'''
