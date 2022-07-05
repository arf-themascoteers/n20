from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
from pandas.api.types import is_string_dtype, is_numeric_dtype
import torch


class SahaDataset(Dataset):
    def __init__(self, is_train):
        self.TRAIN_RATIO = 2
        self.NAN_TOLERANCE = 0.5
        self.is_train = is_train
        self.file_location = "saha.csv"
        csv_data = pd.read_csv("saha.csv")
        self.total = len(csv_data)

        df = pd.DataFrame(csv_data)
        df = self._normalize(df)

        df.to_csv("out.csv")

        self.test_count = len(df) // self.TRAIN_RATIO
        if len(df) % self.TRAIN_RATIO != 0:
            self.test_count += 1
        self.train_count = len(df) - self.test_count
        self.count = self.train_count
        if self.is_train is False:
            self.count = self.test_count

        self.input_map = list()
        self.x_dim = 0
        self.y_dim = 0
        for index, row in df.iterrows():
            last_cell = row[len(row)-1]
            self.y_dim = self._length(last_cell)
            for i in range(len(row)-1):
                cell = row[i]
                length = self._length(cell)
                self.x_dim = self.x_dim + length
                if length == 1:
                    self.input_map.append(df.columns[i])
                else:
                    for counter in range(length):
                        self.input_map.append(df.columns[i]+str(counter))
            break

        self.samples = torch.zeros((self.count, self.x_dim))
        self.targets = torch.zeros(self.count, dtype=torch.float32)

        current_index = 0
        for index, row in df.iterrows():
            mod = index % self.TRAIN_RATIO
            if is_train and mod == 0:
                continue
            if is_train is False and mod != 0:
                continue
            last_cell = row[len(row)-1]
            self.targets[current_index] = last_cell
            start_index = 0
            end_index = 0
            for i in range(len(row)-1):
                cell = row[i]
                if torch.is_tensor(cell):
                    end_index = start_index + cell.shape[0]
                else:
                    end_index = start_index + 1
                self.samples[current_index, start_index:end_index] = cell
                start_index = end_index
            current_index = current_index + 1

    def _length(self, var):
        if torch.is_tensor(var):
            return var.shape[0]
        else:
            return 1

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def _normalize(self, df):
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                df = self._normalize_numeric(df, col)
            elif is_string_dtype(df[col]):
                df = self._normalize_string(df, col)
        return df

    def _normalize_string(self,df, col):
        df[col] = df[col].str.strip()
        uniques = list(df[col].unique())

        for i in range(len(df[col])):
            str = df[col][i]
            value = torch.zeros(len(uniques))
            if str in uniques:
                value[uniques.index(str)] = 1
            df.at[i,col] = value
        return df

    def _normalize_numeric(self,df, col):
        x = df[[col]].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df[col] = x_scaled
        return df

if __name__ == "__main__":
    d = SahaDataset(is_train=True)
    from torch.utils.data import DataLoader
    dl = DataLoader(d, batch_size=20000)
    for x,y in dl:
        print(x.shape)
        print(y)
        print(torch.max(x))
        print(torch.min(x))
        exit(0)


