import pandas as pd
import utils

df = pd.read_csv('datasets/RacismDetectionDataSet.csv')
df = df.rename(columns={"Comment": "input", "Label": "class"})
pos = df.loc[(df['class'] == 1)]
neg = df.loc[(df['class'] == 0)]

def get_inputs(limit, shuffle):
    return utils.return_df(pos, limit, shuffle), utils.return_df(neg, limit, shuffle)