def return_df(d, limit, shuffle):
    if shuffle :
        return d.sample(frac=1).reset_index(drop=True).head(limit)
    return d.head(limit)


