import numpy as np
import pandas as pd

y_pred = list(np.random.uniform(0.4,0.6,2000)) + list(np.random.uniform(0.5,0.7,8000))
y_label = [0] * 2000 + [1] * 8000

def auc(y_pred, y_label):
    pair = list(zip(y_label, y_pred))
    pair = sorted(pair, key= lambda x : x[1])
    df = pd.DataFrame([[x[0],x[1], i] for i, x in enumerate(pair)],columns = ['label', 'pred', 'rank'])

    for k,v in df.pred.value_counts().items():
        if(v == 1):
            continue
        rank_mean = df[df.pred ==k]['rank'].mean()
        df.loc[df.pred ==k, 'rank'] = rank_mean
    
    pos_df = df[df.label == 1]

    m = pos_df.shape[0]
    n = df.shape[0] - m

    return (pos_df["rank"].sum() - m*(m+1)/2) / (m*n)

print(auc(y_pred, y_label))
