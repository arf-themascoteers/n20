import pandas as pd

file_location = "kidney_disease.csv"
csv_data = pd.read_csv(file_location)
df = pd.DataFrame(csv_data)
print(df["pc"].unique())

for i in df["pc"]:
    print(i)