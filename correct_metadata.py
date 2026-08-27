import pandas as pd

df = pd.read_csv(
    "./data/metadata.csv",
    sep='|',
    header=None,
    names=['name', 'text', 'normalized_text']
)

df.to_csv('./data/metadata.csv')