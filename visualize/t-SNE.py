import numpy as np
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data.sampler import SubsetRandomSampler
from fn.checkpoints import CheckpointIO
from Model12 import Model
from Model12_so import Model11
from data.dataloader import ModelNet, ScanNet
from data.dataloader12_sd_so import ModelNet11, ScanObjectNet11, ShapeNet9, ScanObjectNet9

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# Random state.
RS = 20150101 + 1

label_9 = {0: "bag", 1: "bed", 2: "cabinet", 3: "chair", 4: "display",
           5: "pillow", 6: "shelf", 7: "sofa", 8: "table"}

label_10 = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet", 4: "chair",
            5: "lamp", 6: "monitor", 7: "plant", 8: "sofa", 9: "table"}

label_11 = {0: "bed", 1: "cabinet", 2: "chair", 3: "desk", 4: "display",
            5: "door", 6: "shelf", 7: "sink", 8: "sofa", 9: "table", 10: "toilet"}


def scatter(x, colors):
    # data nomalization
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", 20))  # 'Accent', 'hls'
    palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
               '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324',
               '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
               '#ffffff', '#000000']
    # palette = ["#B0C4DE", "#FF00FF", "#1E90FF", "#FA8072", "#EEE8AA", "#FF1493",
    # "#7B68EE", "#FFC0CB", "#696969", "#556B2F", "#CD853F", "#000080", "#32CD32",
    # "#7F007F", "#B03060", "#800000", "#483D8B", "#008000", "#3CB371", "#008B8B",
    # "#FF0000", "#FF8C00", "#FFD700", "#00FF00", "#9400D3", "#00FA9A", "#DC143C",
    # "#00FFFF", "#00BFFF", "#0000FF", "#ADFF2F", "#DA70D6"]

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(len(np.unique(colors))):
        sc = ax.scatter(x[colors == i, 0], x[colors == i, 1], lw=2, s=20, marker='x', c=palette[i], label=label_10[i])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')


if __name__ == '__main__':
    # Arguments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Shorthands
    out_dir = 't_sne'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    batch_size = 4
    test_dataset = ScanNet('/data1/zoulongkun/spacu-master', 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    model = Model(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_io = CheckpointIO('out/myDA12_sup_s', model=model, optimizer=optimizer)
    # checkpoint_io = CheckpointIO()

    try:
        # load_dict = checkpoint_io.load('model_best.pt') # embed_data_m2m_sup
        # load_dict = checkpoint_io.load('model_best(166-58.4).pt')  # embed_data_m2r
        # load_dict = checkpoint_io.load('model_best(66-46.4).pt')  #  embed_data_m112m11_sup
        # load_dict = checkpoint_io.load('model_best(326-71.79).pt')  # embed_data_m112r11
        # load_dict = checkpoint_io.load('model_best(175-55.00).pt')  # embed_data_s92s9_sup
        # load_dict = checkpoint_io.load('model_best(197-60.75).pt')  # embed_data_s92r9
        # load_dict = checkpoint_io.load('model_best(44-42.91).pt')  # embed_data_s2s_sup
        # load_dict = checkpoint_io.load('model_best(53-54.21).pt')  # embed_data_s2r
        load_dict = checkpoint_io.load('model_best(32-48.67).pt')  # embed_data_s2s_sup2
    except FileExistsError:
        print('There is no model to load!!')
        load_dict = dict()

    fea_list = []
    label_list = []

    with torch.no_grad():
        model.eval()
        for data in tqdm(test_loader):
            cloud = data.get('cloud').to(device).float()
            label = data.get('label').long().tolist()
            label_list.append(label)
            fea = model.encoder(cloud)
            fea = model.classifier.mlp1(fea)
            fea = model.classifier.dp1(fea)
            fea = model.classifier.mlp2(fea)
            # fea = logits
            fea_list.append(fea.tolist())
    with open('./t_sne/embed_data_s2s_sup.json', 'w') as f:
        json.dump((sum(fea_list, []), sum(label_list, [])), f)

    emb_filename = "./t_sne/"
    data_name = ['embed_data_s2s_sup.json', 'embed_data_s2s_sup2.json', 'embed_data_s2r.json']
    # data_name = ['embed_data_s92s9_sup.json', 'embed_data_s92r9.json']
    # data_name = ['embed_data_m2m_sup.json', 'embed_data_m2r.json']
    # data_name = ['embed_data_m112m11_sup.json, embed_data_m112r11.json']
    for name in data_name:
        file = emb_filename + name
        data = json.load(open(file, 'r'))
        X = np.array(data[0])
        Y = np.array(data[1])
        # proj = TSNE(perplexity=30, n_iter=15000, n_components=2, learning_rate=300, early_exaggeration=10, init='pca', verbose=1, random_state=RS).fit_transform(X)
        proj = TSNE(random_state=RS + 1).fit_transform(X)
        scatter(proj, Y)
        # plt.legend(fontsize='small', facecolor='white', bbox_to_anchor=(0.9, 0.6), borderaxespad=0)
        plt.legend(fontsize='x-small', facecolor='white', loc=3)
        plt.savefig(file + '.png', dpi=300)
        plt.show()
