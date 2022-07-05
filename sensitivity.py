from ckd_net import CKDNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from apollo_dataset import ApolloDataset
from torchvision import transforms
import matplotlib.pyplot as plt

def sensitivity():
    NUM_EPOCHS = 1
    BATCH_SIZE = 500

    dataset = ApolloDataset(is_train=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = CKDNet(dataset.x_dim)
    model.load_state_dict(torch.load("models/ckd.h5"))
    model.train()
    x_grads = torch.zeros(model.net[0].in_features, requires_grad=False)
    for epoch in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            y_pred = model(data)
            loss = F.nll_loss(y_pred, y_true)
            loss.backward()
            grad_div_weights = (model.net[0].weight.grad.T / model.net[0].weight.T) * (1 - loss)
            grad_div_weights = grad_div_weights.mean(dim=1)
            x_grads = x_grads + grad_div_weights
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')

    mu = x_grads.mean()
    sigma = torch.std(x_grads)
    x_grads = (x_grads - mu)/sigma
    x = list(range(0,len(x_grads)))
    fig, ax = plt.subplots()
    ax.scatter(x,x_grads.detach().numpy())
    for i, txt in enumerate(dataset.input_map):
        ax.annotate(txt, (x[i], x_grads[i]))
    plt.show()


if __name__ == "__main__":
    sensitivity()




