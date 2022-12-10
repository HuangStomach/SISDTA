import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from predict import Predict

def plot_xy(x_values, label, title):
    """绘图"""
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    df['label'] = label
    sns.scatterplot(x="x", y="y", hue="label", data=df)
    plt.title(title)
    plt.show()

if __name__=='__main__':
    predict = Predict()
    predict.predict()

    x = []
    y = []
    for d_index, p_index, d_vecs, p_embeddings, y, classes in predict.loader():
        _, encoded, _, _ = predict.model()(d_index, p_index, d_vecs, p_embeddings, predict.dataset())

        tsne = TSNE(n_components=2)
        x = encoded.detach().numpy()
        y = classes.detach().numpy()

        indice = np.where((y == 231) | (y == 224), True, False)
        xs = tsne.fit_transform(x[indice])
        plot_xy(xs, y[indice], "t-sne")
        break
        # for e, c in zip(encoded, classes):
            # x.append(e)
            # y.append(c)