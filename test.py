from ckd_net import CKDNet
from torch.utils.data import DataLoader
import torch
from apollo_dataset import ApolloDataset


def test():
    BATCH_SIZE = 500
    dataset = ApolloDataset(is_train=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = CKDNet(dataset.x_dim)
    model.load_state_dict(torch.load("models/ckd.h5"))
    model.eval()
    correct = 0
    total = 0
    print(f"Test started ...")
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            pred = torch.argmax(y_pred, dim=1, keepdim=True)
            correct += pred.eq(y_true.data.view_as(pred)).sum()
            total += 1

    print(f"{correct} correct among {len(dataset)}")

if __name__ == "__main__":
    test()
