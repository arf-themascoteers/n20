from n20_net import N20Net
from torch.utils.data import DataLoader
import torch
from saha_dataset import SahaDataset


def test():
    BATCH_SIZE = 2000
    dataset = SahaDataset(is_train=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = N20Net(dataset.x_dim)
    model.load_state_dict(torch.load("models/saha.h5"))
    model.eval()
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = None
    print(f"Test started ...")
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            y_pred = y_pred.reshape(-1)
            loss = criterion(y_pred, y_true)
            print("Ground Truth\t\tPredicted")
            for i in range(y_pred.shape[0]):
                gt_val = y_true[i]
                predicted = y_pred[i]
                print(f"{gt_val:.4f}\t\t\t\t{predicted:.4f}")
    print(f"MSE {loss:.4f}")
if __name__ == "__main__":
    test()
