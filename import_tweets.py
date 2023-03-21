import pandas as pd
import utils

df = pd.read_csv('labeled_data.csv')
df = df.drop(columns=['Unnamed: 0', 'count', 'offensive_language', 'neither', 'hate_speech'])
df = df.rename(columns={"tweet": "input"})
df['class'] = df['class'].map(lambda x: 0 if x == 2 else 1)

pos = df.loc[(df['class'] == 1)]
neg = df.loc[(df['class'] == 0)]

def get_inputs(limit, shuffle):
    return utils.return_df(pos, limit, shuffle), utils.return_df(neg, limit, shuffle)