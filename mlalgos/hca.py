"""Clustering analysis"""

import pickle
from time import perf_counter
from os import path, mkdir, makedirs
from random import randint
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import umap.plot
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import TSNE

from util.util_log import Logger
from os.path import join


class HCA:
    """ HCA cluster.

    Args:
        dataset_names (list): Names for input datasets, e.g., `ori`.
            All the data will be combined for clustering
        grid_size ((int, int)): UIHash grid size
        channel (int): UIHash channel number

    Attributes:
        opt_path (str): Root dir for output. `../output/search` here
        ipt_path (str): Root dir for load input data. `../output/hash` here
        postfix_size (str): A postfix for outputs, indicating UIHash size.
            E.g., '5x5x10'
        postfix_datasets (str): Another postfix for outputs, indicating
            datasets HCA takes as inputs for clustering. E.g., 'ori_re'
        distance_data (numpy array): Distances among inputs
    """
    def __init__(self, dataset_names: list,
                 grid_size: (int, int) = (5, 5),
                 channel: int = 10):
        opt_path = join("..", "output", "search")
        if not path.exists(opt_path):
            makedirs(opt_path)

        self.ipt_path = join("..", "output", "hash")
        self.postfix_size = f"{grid_size[0]}x{grid_size[1]}x{channel}"
        self.hash_list = None
        self.apk_xml_database_list = None
        self.postfix_datasets = '_'.join(dataset_names)
        for n in dataset_names:
            hash_file = join(self.ipt_path, f"hash_{n}_{self.postfix_size}.npy")
            name_file = join(self.ipt_path, f"name_{n}_{self.postfix_size}.npy")
            hash_list = np.load(hash_file, allow_pickle=True)
            self.apk_xml_list = np.load(name_file, allow_pickle=True)
            # add a new column to indicate the dataset a data item belongs to
            col_dataset_name = [n] * len(self.apk_xml_list)
            _apk_xml_list = np.expand_dims(self.apk_xml_list, axis=1)
            apk_xml_dataset_list = np.insert(_apk_xml_list, 1,
                                             values=col_dataset_name, axis=1)
            # each item in the new list: ['apk xml', n]
            if self.hash_list is None:
                self.hash_list = hash_list
            else:
                self.hash_list = np.vstack((self.hash_list, hash_list))
            if self.apk_xml_database_list is None:
                self.apk_xml_database_list = apk_xml_dataset_list
            else:
                self.apk_xml_database_list = \
                    np.vstack((self.apk_xml_database_list,
                               apk_xml_dataset_list))

        self.opt_path = opt_path
        self.opt_subpath = ""
        self.cluster_path = ""
        self.distance_data = None

    @staticmethod
    def hierarchy_cluster(data: np.array, method='average',
                          threshold=5., plot_dendrogram=False,
                          criterion='distance'):
        z = linkage(data, method)
        cluster_assignments = fcluster(z, threshold, criterion=criterion)
        num_clusters = cluster_assignments.max(initial=None)
        indices = HCA.get_cluster_indices(cluster_assignments)
        if plot_dendrogram:
            dendrogram(z)
            plt.show()
        return num_clusters, indices

    @staticmethod
    def get_cluster_indices(cluster_assignments):
        n = cluster_assignments.max()
        indices = []
        for cluster_number in range(1, n + 1):
            indices.append(np.where(cluster_assignments == cluster_number)[0])
        return indices

    def hca(self, distance: str = 'euclidean', method: str = 'average',
            plot_dendrogram: bool = False, threshold: float = 0.8,
            criterion: str = 'distance'):
        """ Run HCA.

        Args:
            distance (str): Distance matric in HCA
            method (str): Method to calculate inter-cluster distances.
            threshold (float): Threshold to merge clusters
            criterion (str): The criterion to use in forming flat clusters
            plot_dendrogram (bool): Draw dendrogram using scipy

        .. Note:: field `distance` can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
            'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
            'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
        """
        brief = f'{criterion}_{distance}_{method}_{threshold}'
        self.opt_subpath = join(self.opt_path,
                                f"{self.postfix_datasets}_{self.postfix_size}")
        if not path.exists(self.opt_subpath):
            makedirs(self.opt_subpath)
        self.cluster_path = join(self.opt_subpath, f"{brief}.npy")
        if path.exists(self.cluster_path):
            print("hca results loaded:", self.cluster_path)
            return
        log = Logger(join(self.opt_subpath, f"{brief}.log"))

        # if data is already there, then just load it
        distance_path = join(self.opt_subpath, f"distance_{distance}.npy")
        load_data = path.exists(distance_path)
        if not load_data:
            log.get_logger.info("calculate distance...")
            t1 = perf_counter()
            size = self.hash_list.shape
            if len(size) == 3:
                # flatten each vector
                self.hash_list = np.reshape(
                    self.hash_list,
                    newshape=(size[0], size[1] * size[2]))
                for i in range(size[0]):
                    self.hash_list[i][0] = self.hash_list[i][0] + 10e-6

            # noinspection PyTypeChecker
            self.distance_data = pdist(self.hash_list, metric=distance)

            np.save(distance_path, self.distance_data, allow_pickle=True)
            t2 = perf_counter()
            log.get_logger.info(
                f"{distance} distance saved in "
                f"{distance_path} ({t2 - t1} s)")
        else:
            if self.distance_data is None:
                log.get_logger.info("loading:", distance_path, end="...")
                self.distance_data = np.load(distance_path, allow_pickle=True)
                log.get_logger.info("done.")

        t3 = perf_counter()
        cnum, indices = \
            self.hierarchy_cluster(self.distance_data, method=method,
                                   threshold=threshold, criterion=criterion,
                                   plot_dendrogram=plot_dendrogram)
        np.save(self.cluster_path, indices, allow_pickle=True)
        t4 = perf_counter()
        log.get_logger.info(f"threshold: {threshold}, clusters: {cnum}, "
                            f"max_cluster_size: {max([len(i) for i in indices])}")
        log.get_logger.info(
            f"hca results saved in {self.cluster_path} ({t4 - t3}s)")

    def distribute_ui_by_cluster(self, cluster_npy: str, name: str,
                                 only_largest_group: bool = False,
                                 hash_path: str,
                                 other_group: int = 0,
                                 group_ids: list = None):
        r = np.load(cluster_npy, allow_pickle=True)
        _opt_path = f'{self.opt_path}clusters_{name}'
        max_length = max([len(i) for i in r])
        if not path.exists(_opt_path):
            mkdir(_opt_path)
        num_cluster = r.shape[0]
        k = 0

        if not group_ids:
            if not only_largest_group:
                clusters = range(num_cluster)
            else:
                clusters = [
                    i for i in range(num_cluster) if len(
                        r[i]) == max_length]
                clusters.extend([randint(0, num_cluster - 1)
                                 for _ in range(other_group)])
                clusters = list(set(clusters))

        else:
            clusters = group_ids
        for i in clusters:
            __opt_path = f'{_opt_path}/{i}'
            if not path.exists(__opt_path):
                mkdir(__opt_path)
            for index in r[i]:
                apk, xml = self.apk_xml_list[index].split(' ')
                xml = xml.replace('.xml', '')
                copyfile(
                    f'{hash_path}{apk}/{xml}.xml',
                    f'{__opt_path}/{apk}@{xml}.xml')
                copyfile(
                    f'{hash_path}{apk}/{xml}.jpg',
                    f'{__opt_path}/{apk}@{xml}.jpg')
                k += 1
                print(k)

    def scatter_tsne(self, cluster_npy: str, dist_npy: str,
                     save_path: str, calculate: bool = False,
                     precomputed_dist: bool = False,
                     tsne_perplexity: int = 5,
                     tsne_lr: float = 200,
                     tsne_iter: int = 1000):
        """ Draw scatter figure for t-SNE visulization

        """
        if (not path.exists(save_path)) or calculate:
            print('start to calculate tsne')
            if precomputed_dist:
                dist = np.load(dist_npy, allow_pickle=True)
                dist = squareform(dist, force='tomatrix', checks=True)
                tsne = TSNE(metric='precomputed',
                            perplexity=tsne_perplexity,
                            learning_rate=tsne_lr,
                            n_iter=tsne_iter,
                            n_components=2)
                tsne.fit_transform(dist)
                a = tsne.embedding_
                np.save(save_path, a, allow_pickle=True)
            else:
                tsne = TSNE(metric='euclidean',
                            perplexity=tsne_perplexity,
                            learning_rate=tsne_lr,
                            n_iter=tsne_iter,
                            n_components=2)
                size = self.hash_list.shape
                if len(size) == 3:
                    self.hash_list = np.reshape(
                        self.hash_list, newshape=(size[0], size[1] * size[2]))
                tsne.fit_transform(self.hash_list)
                a = tsne.embedding_
                np.save(save_path, a, allow_pickle=True)

        tsne_cord = np.load(save_path, allow_pickle=True)
        indices = np.load(cluster_npy, allow_pickle=True)
        _x, _y = tsne_cord[:, 0], tsne_cord[:, 1]
        _cluster = [0] * len(_x)
        for idx, samples in enumerate(indices):
            for s in samples:
                _cluster[s] = idx

        _data = np.dstack((_x, _y, _cluster))[0]
        data = pd.DataFrame(_data,
                            columns=['x', 'y', 'cluster'],
                            index=[i for i in range(len(tsne_cord))])
        sns.scatterplot(x='x', y='y', hue='cluster', data=data,
                        palette='Set2', legend='brief', size=0.4)
        plt.show()

    def scatter_umap(self, cluster_npy: str = "",
                     random_seed: int = 4, n_neighbor: int = 15,
                     min_dist: float = 0.1, metric: str = "euclidean",
                     use_umap_plotter: bool = True,
                     benign_path: list = None,
                     assigned_cluster: list = None,
                     width: int = 800, height: int = 800,
                     xyrange: (int, int, int, int) = None):
        """ Draw scatter figure for UMAP visulization

        """
        if len(cluster_npy) == 0:
            cluster_npy = self.cluster_path
        save_file_path = join(self.opt_subpath,
                              f"umap_{metric}_{min_dist}_{random_seed}_{n_neighbor}.pickle")
        indices = np.load(cluster_npy, allow_pickle=True)
        use_umap_update = True
        benign_emdedding = None
        benign_hash, benign_name = None, None
        mapper = None
        umap_input_data = None
        umap_input_label_ori = None

        if path.exists(save_file_path):
            # load mapper if exists
            with open(save_file_path, 'rb') as f:
                mapper = pickle.load(f)
            print("umap embedding loaded:", save_file_path)

        if benign_path is not None:
            # load and append extra data
            benign_hash = np.load(benign_path[0], allow_pickle=True)
            benign_name = np.load(benign_path[1], allow_pickle=True)  # pkg xml
            if use_umap_update and mapper is not None:
                benign_emdedding = mapper.transform(benign_hash)
            else:
                umap_input_data = np.vstack((self.hash_list, benign_hash))
                umap_input_label_ori = np.hstack((self.apk_xml_database_list, benign_name))
            print("update mapper done,", len(benign_name), "data added")
        else:
            umap_input_data = self.hash_list
            umap_input_label_ori = self.apk_xml_database_list

        umap_input_label: list = [0] * len(umap_input_label_ori)

        for i, label in enumerate(umap_input_label_ori):
            apk, xml = label[0].split(' ')
            umap_input_label[i] = f"APK: {apk}\tXML: {xml}\tSRC: {label[1]}"

        if mapper is None:
            # run umap embedding
            print("umap embedding start...")
            size = umap_input_data.shape
            if len(size) == 3:
                umap_input_data = np.reshape(
                    umap_input_data, newshape=(size[0], size[1] * size[2]))

            t1 = perf_counter()
            mapper = umap.UMAP(random_state=random_seed,
                               metric=metric,
                               n_neighbors=n_neighbor,
                               min_dist=min_dist,
                               low_memory=True).fit(umap_input_data)
            t2 = perf_counter()
            print(f"umap embedding generated in ({t2 - t1} s)")
            with open(save_file_path, mode='wb') as f:
                pickle.dump(mapper, f)
            print("mapper saved done in", save_file_path)

        cluster = [0] * umap_input_data.shape[0]
        for idx, samples in enumerate(indices):
            for s in samples:
                cluster[s] = idx

        if not use_umap_plotter:
            embedding = mapper.transform(umap_input_data)
            _x, _y = embedding[:, 0], embedding[:, 1]
            _dataf = np.dstack((_x, _y, cluster))[0]
            data = pd.DataFrame(_dataf,
                                columns=['x', 'y', 'cluster'],
                                index=[i for i in range(len(embedding))])
            sns.scatterplot(x='x', y='y', hue='cluster', data=data,
                            palette='Set2', legend='brief', size=0.4)
            plt.show()
        else:
            new_cluster = cluster.copy()
            if benign_name is not None:
                new_cluster.extend([-100] * len(benign_name))
            if benign_emdedding is not None:
                all_embedding_ = np.vstack(
                    (mapper.embedding_, benign_emdedding))
            else:
                all_embedding_ = mapper.embedding_
            hover_data = pd.DataFrame({"index": np.arange(len(umap_input_label)),
                                       "label": umap_input_label})

            if xyrange is not None:
                assigned_cluster = []
                x1, x2, y1, y2 = xyrange
                banned_indices = []
                for i, e in enumerate(new_cluster):
                    xa, ya = all_embedding_[i, 0], all_embedding_[i, 1]
                    if not x1 < xa < x2:
                        continue
                    if not y1 < ya < y2:
                        continue
                    banned_indices.append(i)
                x = [True if i in banned_indices else False for i in range(len(new_cluster))]
                new_cluster = np.array(new_cluster)[x].tolist()
                label = np.array(umap_input_label)[x].tolist()
                all_embedding_ = all_embedding_[x]
                hover_data = pd.DataFrame({'index': np.arange(len(label)),
                                           'label': label})

            if assigned_cluster is not None:
                x = [True if i in assigned_cluster else False for i in new_cluster]
                new_cluster = [i for i in new_cluster if i in assigned_cluster]
                label = [i for i in umap_input_label if i in assigned_cluster]
                all_embedding_ = all_embedding_[x]
                hover_data = pd.DataFrame({'index': np.arange(len(label)),
                                           'label': label})

            from util.util_draw import interactive
            if benign_name is not None:
                p = interactive(all_embedding_, labels=np.array(new_cluster),
                                hover_data=hover_data, len_black=len(benign_emdedding))
            else:
                p = interactive(all_embedding_, labels=np.array(new_cluster),
                                hover_data=hover_data, height=width, width=height)
            umap.plot.show(p)
            
            # the bokeh svg figure can only be saved by api
            from bokeh.io import export_svg
            try:
                p.output_backend = "svg"
                file_name = join(self.opt_subpath, f"plot{perf_counter()}.svg")
                export_svg(p, filename=file_name)
            except RuntimeError:
                print("svg is not exported: webdriver missing in PATH")

    def save_cluster_indices(self):
        """ output cluster results into a text file `clusters.txt`
        """
        cluster_indices = np.load(self.cluster_path, allow_pickle=True)
        cluster = [0] * len(self.apk_xml_database_list)
        for idx, samples in enumerate(cluster_indices):
            for s in samples:
                cluster[s] = idx
        lines = []
        for i in range(len(cluster)):
            apk, xml = self.apk_xml_database_list[i][0].split(' ')
            src = self.apk_xml_database_list[i][1]
            line = f"{cluster[i]} {apk} {xml} {src}\n"
            lines.append(line)
        with open(join(self.opt_subpath, "clusters.txt"), mode='w+',
                  encoding='utf-8') as f:
            f.writelines(lines)


if __name__ == "__main__":
    d, m = 'euclidean', 'ward'
    c = 'distance'
    t = 10
    hca = HCA(dataset_names=["ori", "re"])

    hca.hca(threshold=t, distance=d,
            method=m, criterion=c)
    seed = 42
    umap_n_neighbor = 2000
    umap_min_dist = 0.5
    umap_metric = 'euclidean'
    hca.save_cluster_indices()
