from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
from pandas.api.types import is_string_dtype, is_numeric_dtype
import torch
from sklearn import model_selection


class SahaDataset(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        self.file_location = "saha.csv"
        csv_data = pd.read_csv("saha.csv")
        self.total = len(csv_data)
        df = pd.DataFrame(csv_data)
        df = self._preprocess(df)

        # Dumping for debug purpose
        df.to_csv("out.csv")

        train, test = model_selection.train_test_split(df, test_size=0.2)
        df = train
        if not self.is_train:
            df = test

        self.x_tensor, self.y_tensor = self._tensorise(df)
        self.x_dim = self.x_tensor.shape[1]

    def _calculate_dim(self, x):
        dim = 0
        for index, row in x.iterrows():
            for i in range(len(row)):
                cell = row[i]
                length = self._length(cell)
                dim = dim + length
            break
        return dim

    def _tensorise(self, df):
        x = df.iloc[:, :-1]
        x_dim = self._calculate_dim(x)
        last_column = df.iloc[: , -1]
        count = len(df)
        tensor_x = torch.zeros((count, x_dim))

        row_index = 0
        for index, row in x.iterrows():
            start_index = 0
            end_index = 0
            for i in range(len(row)):
                cell = row[i]
                if torch.is_tensor(cell):
                    end_index = start_index + cell.shape[0]
                else:
                    end_index = start_index + 1
                tensor_x[row_index, start_index:end_index] = cell
                start_index = end_index
            row_index = row_index + 1
        last_column = list(last_column)
        tensor_y = torch.Tensor(last_column)
        return tensor_x, tensor_y

    def _length(self, var):
        if torch.is_tensor(var):
            return var.shape[0]
        else:
            return 1

    def __len__(self):
        return self.x_tensor.shape[0]

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]

    def _preprocess(self, df):
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


