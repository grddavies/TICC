# %% Imports & set Matplotlib scheme
import numpy as np
from networkx import nx
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
from ticclib import TICC
from ticclib.testing import RandomData, best_f1
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.style.use('dark_background')
colors = plt.rcParams["axes.prop_cycle"]()


def c_inv(colour):
    """Invert input colour hexcode"""
    colour = int(colour[1:], 16)
    comp_colour = 0xFFFFFF ^ colour
    comp_colour = "#%06X" % comp_colour
    return comp_colour


# %% Generate data
n_features = 5
label_seq = [0, 1, 2, 0, 2, 1]
samples_per_segment = 200
window_size = 8

# Derived from above params
k = len(set(label_seq))  # Num clusters
t = samples_per_segment*len(label_seq)  # total ts length
breaks = [i*t//len(label_seq) for i in range(1, len(label_seq) + 1)]
palette = {n: c['color'] for n, c in zip(range(n_features), colors)}
randomdata = RandomData(seed=1234, n_features=n_features,
                        window_size=window_size)
X, y_true = randomdata.generate_points(label_seq, breaks)


# %% Plot Synthetic Data
def plot_synthetic_data(X, break_points):
    fig, axes = plt.subplots(5, sharex=True, sharey=True, figsize=(14, 6))
    for i, ax in enumerate(axes):
        ax.plot(X[:, i], color=palette[i], linewidth=0.75,
                label=f"feature {i}")
        ax.get_yaxis().set_visible(False)
        ax.legend(loc="upper left")
        for p in range(len(break_points)-1):
            start, end = break_points[p], break_points[p+1]
            width = end - start
            rect = Rectangle((start, -3), width, 6, alpha=0.2,
                             facecolor='white')
            if p % 2 == 0:
                ax.add_patch(rect)
            # ax.vlines(p, 0, 1, transform=ax.get_xaxis_transform(),
            #           color='white'
            #           )
        handles, labels = ax.get_legend_handles_labels()
    plt.subplots_adjust(hspace=.0)
    plt.xlabel("time samples")
    plt.suptitle("Synthetic Multivariate Timeseries", y=0.94)


plot_synthetic_data(X, breaks)

# %% Fit TICC and GMM to data
scaler = StandardScaler()
X = scaler.fit_transform(X)
ticc = TICC(n_clusters=k, window_size=window_size, random_state=1234, beta=200)
gmm = GaussianMixture(n_components=k, random_state=1234)
X_stacked = ticc.stack_data(X)

y_ticc = ticc.fit(X).labels_
y_gmm = gmm.fit_predict(X_stacked)

# %% Macro F1 Scores
f1_ticc = f1_score(y_true, y_ticc, average='micro')
f1_gmm = f1_score(y_true, y_gmm, average='micro')
print(f"TICC F1 score = {f1_ticc}\n GMM F1 score = {f1_gmm}")

# %% Plot Cluster Assignments
fig, axes = plt.subplots(3, sharex=True, figsize=(14, 8))
axes[0].plot(y_true, color=palette[0], label='Ground Truth')
axes[1].plot(y_ticc, color=palette[1], label='TICC')
axes[2].plot(y_gmm, color=palette[2], label='GMM')
# axes[3].plot(y_spc, color=palette[3], label='Spectral Clustering')
for ax in axes:
    ax.legend(loc='upper left')


# %% Plot Markov Random Fields
def plot_MRF(adj_mat, thresh=0.05, ax=None):
    # Filter out subthreshold values in adj_mat
    adj_mat = np.where(
        np.abs(adj_mat) > thresh,
        adj_mat,
        np.zeros(adj_mat.shape)
        )
    # Create graph object
    G = nx.Graph(adj_mat)
    pos = nx.layout.circular_layout(G, scale=0.5)
    node_colors = [palette[n] for n in G.nodes]
    edge_colors = [v['weight'] for _, _, v, in G.edges(data=True)]
    vmin, vmax = min(edge_colors), max(edge_colors)
    cmap = plt.get_cmap()
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=700)
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors,
                           edge_cmap=cmap,
                           edge_vmin=vmin,
                           edge_vmax=vmax,
                           width=2.5)
    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmax, vmax=vmin)
                               )
    fig.colorbar(sm, orientation="horizontal", pad=0.01, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)


def compare_MRFs(ticc_num, gt_num, w):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.1))
    plot_MRF(ticc.clusters_[ticc_num].split_theta(window_size)[0], ax=ax[0])
    ax[0].set_title(f"TICC Cluster {ticc_num}", size='small')
    plot_MRF(randomdata.clusters[gt_num][w:(w+1)*n_features, :n_features],
             ax=ax[1])
    ax[1].set_title(f"Ground Truth Cluster {gt_num}", size='small')
    wtype = ("Intra" if w == 0 else "Cross")
    title = f"{wtype}-time Correlation Structures"
    plt.suptitle(title, y=1)


compare_MRFs(0, 0, 0)
compare_MRFs(1, 1, 0)

# %% Relabel TICC cluster output and recalculate F1
swap = {0: 1, 1: 0, 2: 2}
y_ticc2 = np.vectorize(swap.get)(y_ticc)
f1_ticc2 = f1_score(y_true, y_ticc2, average='macro')
print(f"New TICC F1 score = {f1_ticc2}")


# %%
def plot_precision_matrices(matrices: list):
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    for i, ax in enumerate(axes):
        im = ax.matshow(matrices[i])
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Cluster {i}", pad=15)


plot_precision_matrices([randomdata.clusters[0], ticc.clusters_[1].MRF_])
