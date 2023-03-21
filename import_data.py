import pandas as pd

# def clean_data(x):

df = pd.read_csv('labeled_data.csv')
df = df.drop(columns=['Unnamed: 0', 'count', 'offensive_language', 'neither', 'hate_speech'])
df = df.rename(columns={"tweet": "input"})
# df.apply(lambda x: 0 if x[0] === 0 else x[0])
df['class'] = df['class'].map(lambda x: 0 if x == 2 else 1)
# df['input'] = df['input'].map(lambda x: 0 if x == 2 else 1)

# df = df.filter(lambda x: 0 if x == 2 else 1)
pos = df.loc[(df['class'] == 1)]
neg = df.loc[(df['class'] == 0)]

# print(list(df.columns.values))
# print(neg.head(10).to_string())


def pos_input(limit):
    return pos.head(limit)

def neg_input(limit):
    return neg.head(limit)