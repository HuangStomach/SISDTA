import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

drug = np.loadtxt('output/davis/sis_ecfps_sim.csv', delimiter=',')
with open('data/davis/ligands_iso.json', 'r') as f:
    drug_names = list(json.load(f).keys())
tsne_sis = TSNE(n_components=2, learning_rate=100).fit_transform(drug)

drug = np.loadtxt('output/davis/default_ecfps_sim.csv', delimiter=',')
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(drug)
 
# 使用PCA 进行降维处理
# pca = PCA().fit_transform(drug)
# 设置画布的大小
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(tsne_sis[:, 0], tsne_sis[:, 1])
plt.subplot(122)
plt.scatter(tsne[:, 0], tsne[:, 1])
for i in range(len(tsne)):
    plt.annotate(drug_names[i], xy = (tsne[i][0], tsne[i][1]), xytext = (tsne[i][0]+0.1, tsne[i][1]+0.1))
# plt.colorbar()
plt.show()
