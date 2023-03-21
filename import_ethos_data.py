import pandas as pd
import utils

df = pd.read_csv(filepath_or_buffer='datasets/Ethos_Dataset_Binary.csv', delimiter=';')
df = df.rename(columns={"comment": "input", "isHate": "class"})

def get_inputs(limit, shuffle, threshold):
    df['class'] = df['class'].map(lambda x: 1 if x > threshold else 0)
    return utils.return_df(df.loc[(df['class'] == 1)], limit, shuffle), utils.return_df(df.loc[(df['class'] == 0)], limit, shuffle)