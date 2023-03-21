import pandas as pd


df = pd.read_csv('labeled_data.csv')
df = df.drop(columns=['Unnamed: 0', 'count', 'offensive_language', 'neither', 'hate_speech'])
df = df.rename(columns={"tweet": "input"})
df['class'] = df['class'].map(lambda x: 0 if x == 2 else 1)

pos = df.loc[(df['class'] == 1)]
neg = df.loc[(df['class'] == 0)]


def return_df(d, limit, shuffle):
    if shuffle :
        return d.sample(frac=1).reset_index(drop=True).head(limit)
    return d.head(limit)
def pos_input(limit, shuffle):
    return return_df(pos, limit, shuffle)

def neg_input(limit, shuffle):
    return return_df(neg, limit, shuffle)
