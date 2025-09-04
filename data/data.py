import pandas as pd

# Download dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Save to CSV for DVC tracking
df.to_csv("adult_income.csv", index=False)

print("Dataset saved as adult_income.csv with shape:", df.shape)
