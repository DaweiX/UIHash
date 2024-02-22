"""Generate input dataset for machine learning"""

from time import perf_counter
from os import makedirs
from os.path import exists, join
from random import randint, shuffle
from typing import Tuple
import os.path

import numpy as np
from torch.utils.data import Dataset
from zipfile import BadZipFile, ZipFile
from os import walk
from xml.dom.minidom import parseString
from androguard.core.bytecodes.axml import AXMLPrinter
import torch
import sys
import argparse


class WildDataSet(Dataset):
    """Dataset for an unlabelled app set, used for detecting only"""

    def __init__(self, dataset_name: str, transform=None,
                 reshape: bool = False,
                 hash_size: Tuple[int, int, int] = (10, 5, 5),
                 siamese_model=None, threshold: float = 0.6):

        size_str = f"{hash_size[1]}x{hash_size[2]}x{hash_size[0]}"
        base_path = join(os.path.abspath(os.path.dirname(__file__)),
                         "..", "output")
        npzfile_raw = join(base_path, "dataset",
                           f"{dataset_name}_{size_str}_raw.npz")
        self.pairs_path = join(base_path, "dataset",
                               f"{dataset_name}_{size_str}_pairs.npy")
        self.dataset_name = dataset_name
        self.transform = transform
        self.threshold = threshold

        if exists(npzfile_raw):
            self.raw_data = np.load(npzfile_raw, allow_pickle=True)['data']
            self.raw_name = np.load(npzfile_raw, allow_pickle=True)['name']
        else:
            # generate ui and name set
            self.model = siamese_model
            data_path = join(base_path, "hash",
                             f"hash_{dataset_name.lower()}_{size_str}.npy")
            name_path = join(base_path, "hash",
                             f"name_{dataset_name.lower()}_{size_str}.npy")

            data_path = data_path.replace("__", "_")
            name_path = name_path.replace("__", "_")
            self.all_data = np.load(data_path, allow_pickle=True)
            self.all_name = np.load(name_path, allow_pickle=True)
            if reshape:
                size = self.all_data[:, 0].shape
                if len(size) < 3:
                    self.all_data = \
                        [np.reshape(i, hash_size) for i in self.all_data]
            self.raw_data = []
            self.raw_name = []
            _t1 = perf_counter()
            print("start filtering")
            self.filter_ui_in_app()
            np.savez(npzfile_raw, data=self.raw_data, name=self.raw_name)
            _t2 = perf_counter()
            print(f"filter similar uis in each app in {_t2 - _t1} s")

        if exists(self.pairs_path):
            self.pairs = np.load(self.pairs_path, allow_pickle=True)
        else:
            # generate ui pairs
            self.generate_pairs()
            print(f"{len(self.pairs)} ui pairs in total")

    def generate_pairs(self):
        """
        to save space, we only save indexes for pairs
          format: (index_a (int), index_b (int))
        """
        self.pairs = []
        print("generating pairs...")
        for i in range(len(self.raw_name) - 1):
            print(i)
            app1 = self.raw_name[i].split(' ')[0]
            for j in range(i + 1, len(self.raw_name)):
                app2 = self.raw_name[j].split(' ')[0]
                if app2 == app1:
                    # we don't compare uis in one app
                    continue
                self.pairs.append((i, j))
        np.save(self.pairs_path, self.pairs, allow_pickle=True)

    def filter_ui_in_app(self):
        # when generating uihash, all the xmls
        # for one app are arranged together, one after another
        xmls_in_apps = dict()
        remove_total = 0
        for i, a in enumerate(self.all_name):
            app = a.split(' ')[0]
            if app not in xmls_in_apps:
                xmls_in_apps[app] = [i]
            else:
                xmls_in_apps[app].append(i)
                
        # make pair-wise ui comparision in each app
        app_total = len(xmls_in_apps.keys())
        for k, app in enumerate(xmls_in_apps.keys()):
            print(f"({k + 1}/{app_total}) {app}")
            xml_indexes = [x for x in xmls_in_apps[app]]
            remove_set = set()
            for i in range(len(xml_indexes) - 1):
                if i in remove_set:
                    continue
                for j in range(i + 1, len(xml_indexes)):
                    ui1 = self.all_data[xml_indexes[i]]
                    ui2 = self.all_data[xml_indexes[j]]
                    ui1, ui2, _ = self.model.deal_data((torch.from_numpy(ui1),
                                                        torch.from_numpy(ui2),
                                                        torch.tensor(-1)))
                    ui = torch.stack((ui1, ui2), 0)
                    ui = ui.unsqueeze(1)
                    o = self.model.net(ui)
                    row = int(o.shape[0] / 2)
                    o1, o2 = o[:row, :], o[row:, :]
                    o1 = torch.squeeze(o1, 0)
                    o2 = torch.squeeze(o2, 0)
                    distance = torch.cosine_similarity(o1, o2).item()
                    if distance > self.threshold:
                        remove_set.add(xml_indexes[j])
                        remove_total += 1
            xml_indexes_new = set(xml_indexes).difference(remove_set)
            self.raw_data.extend([self.all_data[x] for x in xml_indexes_new])
            self.raw_name.extend([self.all_name[x] for x in xml_indexes_new])

        print("len of filtered data:", len(self.raw_data))
        print("xml removed:", remove_total)

    def __getitem__(self, idx: int):
        idx1, idx2 = self.pairs[idx]
        i1, i2 = self.raw_data[idx1], self.raw_data[idx2]
        if self.transform:
            i1 = self.transform(i1)
            i2 = self.transform(i2)
        return i1, i2, torch.from_numpy(np.array(-1))

    def __len__(self):
        return len(self.pairs)

    def get_info(self, idx: int) -> Tuple[str, str, str, str]:
        """ given pair index, return: app1, app2, xml1, xml2 """
        idx1, idx2 = self.pairs[idx]
        n1, n2 = self.raw_name[idx1], self.raw_name[idx2]

        def get_app_xml(name: str):
            app = name.split(' ')[0]
            xml = name.replace(app, '').strip()
            return app, xml

        app1, xml1 = get_app_xml(n1)
        app2, xml2 = get_app_xml(n2)
        return app1, app2, xml1, xml2


class GenPairsForRepack:
    def __init__(self, apk_folders: list, list_txt: str,
                 hash_size: str, npz_prefix: str = "Re"):
        # apk file hash <-> package name, split by a space
        hash_name_npy_path = join(os.path.abspath(os.path.dirname(__file__)),
                                  "..", "output", "hash")
        map_npy = join(hash_name_npy_path, "apk2pkg.npy")
        if not exists(map_npy):
            apk2package_name(apk_folders, map_npy)
        _map = np.load(map_npy)
        self.pkg2apk = dict()
        self.apk2pkg = dict()
        for _m in _map:
            apk, pkg = _m.split(' ')
            apk = apk[:-4]
            self.pkg2apk[pkg] = apk
            self.apk2pkg[apk] = pkg

        # original apk <-> repackage apk, split by a comma
        with open(list_txt, mode='r') as f:
            self._pair = f.readlines()

        self.hash_data_ori = np.load(join(hash_name_npy_path,
                                          f"hash_ori_{hash_size}.npy"))
        self.name_data_ori = np.load(join(hash_name_npy_path,
                                          f"name_ori_{hash_size}.npy"))
        self.hash_data_re = np.load(join(hash_name_npy_path,
                                         f"hash_re_{hash_size}.npy"))
        self.name_data_re = np.load(join(hash_name_npy_path,
                                         f"name_re_{hash_size}.npy"))

        # flatten uihash
        size = self.hash_data_ori.shape
        if len(size) == 3:
            self.hash_data_ori = np.reshape(
                self.hash_data_ori, newshape=(
                    size[0], size[1] * size[2]))
        size = self.hash_data_re.shape
        if len(size) == 3:
            self.hash_data_re = np.reshape(
                self.hash_data_re, newshape=(
                    size[0], size[1] * size[2]))

        self.output_npy_path = join(os.path.abspath(os.path.dirname(__file__)),
                                    "..", "output", "dataset")
        if not exists(self.output_npy_path):
            makedirs(self.output_npy_path)
        self.hash_size = hash_size
        self.npz_prefix = npz_prefix

    def gen_sim_pairs(self) -> int:
        data_npy = join(self.output_npy_path,
                        f"{self.npz_prefix}SP_{self.hash_size}.npy")
        if exists(data_npy):
            print("sim pair data already exists")
            return -1
        data = list()
        repackage_pkgs_dict = dict()
        k = 0
        for i, name in enumerate(self.name_data_ori):
            ori_pkg, ori_xml = name.split(' ')
            if ori_pkg not in repackage_pkgs_dict:
                # 1. consider apks with the
                # same package name in original set
                for j in [idx for idx, i in enumerate(self.name_data_ori)
                          if i == f'{ori_pkg} {ori_xml}']:
                    if j == i:
                        continue
                    data.append([ori_pkg, ori_pkg, ori_xml,
                                 self.hash_data_ori[i], self.hash_data_re[j]])
                    k += 1
                    if k % 200 == 0:
                        print(k)

                # 2. repackage apks defined by Repack

                if ori_pkg not in self.pkg2apk:
                    continue
                ori_apk = self.pkg2apk[ori_pkg]  # hash apk name in Repack
                repackage_apks = [i.split(',')[1]
                                  for i in self._pair if i.startswith(ori_apk)]
                repackage_apks = [i.replace('\n', '') for i in repackage_apks]
                repackage_pkgs = [self.apk2pkg[i] for i in repackage_apks if i in self.apk2pkg]
                repackage_pkgs_dict[ori_pkg] = repackage_pkgs
            else:
                repackage_pkgs = repackage_pkgs_dict[ori_pkg]
                re_data = []
                for re_pkg in repackage_pkgs:
                    # if the xml name in the repackaged app is the same
                    # as the original one, we treat them as similiar uis
                    re_data.extend([idx for idx, i in enumerate(
                        self.name_data_re) if i == f'{re_pkg} {ori_xml}'])
                for j in re_data:
                    re_pkg = self.name_data_re[j].split(' ')[0]
                    # pkg_ori, pkg_re, xml, hash_ori, hash_re
                    data.append([ori_pkg, re_pkg, ori_xml,
                                 self.hash_data_ori[i], self.hash_data_re[j]])
                    k += 1
                    if k % 200 == 0:
                        print(k)

        np.save(data_npy, data)
        print(len(data), "similiar pairs saved done!")
        return len(data)

    def gen_sim_pair_list(self, data_npy: str):
        if exists(data_npy):
            print("sim pair list already exists")
            return
        data = list()
        repackage_pkgs_dict = dict()
        k = 0
        for i, name in enumerate(self.name_data_ori):
            ori_pkg, ori_xml = name.split(' ')
            if ori_pkg not in repackage_pkgs_dict:
                # repackage apks defined by Repack
                ori_apk = self.pkg2apk[ori_pkg]
                repackage_apks = [i.split(',')[1]
                                  for i in self._pair if i.startswith(ori_apk)]
                repackage_apks = [i.replace('\n', '') for i in repackage_apks]
                repackage_pkgs = [self.apk2pkg[i] for i in repackage_apks]
                repackage_pkgs_dict[ori_pkg] = repackage_pkgs
            else:
                repackage_pkgs = repackage_pkgs_dict[ori_pkg]
                re_data = []
                for re_pkg in repackage_pkgs:
                    # if the xml name in the repackaged app is the same
                    # as the original one, we treat them as similiar uis
                    re_data.extend([idx for idx, i in enumerate(
                        self.name_data_re) if i == f'{re_pkg} {ori_xml}'])
                for j in re_data:
                    re_pkg = self.name_data_re[j].split(' ')[0]
                    # pkg_ori, pkg_re, xml, hash_ori, hash_re
                    data.append([f'{ori_pkg}/{ori_xml}', f'{re_pkg}/{ori_xml}'])
                    k += 1
                    if k % 200 == 0:
                        print(k)

        np.save(data_npy, data)
        print(len(data), 'similiar pairs saved done!')

    def gen_unsim_pairs(self, num: int):
        data_npy = join(self.output_npy_path,
                        f"{self.npz_prefix}DP_{self.hash_size}.npy")
        data = list()
        combine_name = np.hstack((self.name_data_ori, self.name_data_re))
        combine_hash = np.vstack((self.hash_data_ori, self.hash_data_re))
        total = len(combine_name)
        k = 0
        while k < num:
            i = randint(0, total - 1)
            j = randint(0, total - 1)
            pkg1, xml1 = combine_name[i].split(' ')
            pkg2, xml2 = combine_name[j].split(' ')
            # skip uis in same package or with same name
            if pkg1 == pkg2:
                continue
            if xml1 == xml2:
                continue
            # besides, we'd better make the pair work with
            # two apps which are not a ori-re pair
            try:
                apk1, apk2 = self.pkg2apk[pkg1], self.pkg2apk[pkg2]
            except KeyError:
                continue
            if f'{apk1},{apk2}' in self._pair:
                continue
            if f'{apk2},{apk1}' in self._pair:
                continue

            # one another item for unsimiliar pair
            data.append([pkg1, pkg2, xml1, xml2,
                         combine_hash[i], combine_hash[j]])
            k += 1
            if k % 200 == 0:
                print(k)

        np.save(data_npy, data)
        print(len(data), 'not similiar pairs saved done!')

    def gen_unsim_pair_list(self, data_npy: str, num: int):
        data = list()
        combine_name = np.hstack((self.name_data_ori, self.name_data_re))
        total = len(combine_name)
        k = 0
        while k < num:
            i, j = randint(0, total - 1), randint(0, total - 1)
            pkg1, xml1 = combine_name[i].split(' ')
            pkg2, xml2 = combine_name[j].split(' ')
            # skip uis in same package or with same name
            if pkg1 == pkg2:
                continue
            if xml1 == xml2:
                continue
            # besides, we'd better make the pair work with
            # two apps which are not a ori-re pair
            try:
                apk1, apk2 = self.pkg2apk[pkg1], self.pkg2apk[pkg2]
            except KeyError:
                continue
            if f'{apk1},{apk2}' in self._pair:
                continue
            if f'{apk2},{apk1}' in self._pair:
                continue

            # one another item for unsimiliar pair
            data.append([f'{pkg1}/{xml1}', f'{pkg2}/{xml2}'])
            k += 1
            if k % 200 == 0:
                print(k)

        np.save(data_npy, data)
        print(len(data), 'not similiar pairs saved done!')


def apk2package_name(apk_folders: list, opt_npy: str):
    def _apk2package_name(_apk_path: str) -> str or None:
        _file = "AndroidManifest.xml"
        try:
            with ZipFile(_apk_path, 'r') as a:
                text = AXMLPrinter(a.read(_file))
                xml = text.get_xml()
                dom = parseString(xml)
                _package_name = dom.getElementsByTagName(
                    'manifest')[0].getAttribute('package')
                return _package_name
        except BadZipFile as e:
            print(e)
            return None

    package_names = set()
    map_list = []
    k = 0
    for apk_folder in apk_folders:
        for dirpath, _, filenames in walk(apk_folder):
            for apk_name in filenames:
                apk_path = join(dirpath, apk_name)
                package_name = _apk2package_name(apk_path)
                pair = f'{apk_name} {package_name}'
                map_list.append(pair)
                package_names.add(package_name)
                k += 1
                print(k, pair)

    np.save(opt_npy, map_list)

    print('diff', len(map_list) - len(package_names))


class LabelledDataSet(Dataset):
    """
    Dataset (for labelled only)

    For the 1st run, the hash npy and name npy will be fit
    into exactly what torch needs as its input, and the results
    will be stored in the same path within time O(n^2). Later,
    the instances of the class will just load the generated data.
    """

    def __init__(self, transform=None, reshape: bool = False,
                 hash_size: Tuple[int, int, int] = (10, 5, 5),
                 npz_prefix: str = "Re", shuffle_data: bool = True):

        size_str = f"{hash_size[1]}x{hash_size[2]}x{hash_size[0]}"
        out_path = join(os.path.abspath(os.path.dirname(__file__)),
                        "..", "output", "dataset")
        npzfile = join(out_path, f"{npz_prefix}_{size_str}.npz")
        self.transform = transform
        if exists(npzfile):
            self.data = np.load(npzfile, allow_pickle=True)['data']
            self.name = np.load(npzfile, allow_pickle=True)['name']
        else:
            sim_path = join(out_path, f"{npz_prefix}SP_{size_str}.npy")
            usim_path = join(out_path, f"{npz_prefix}DP_{size_str}.npy")
            sim = np.load(sim_path, allow_pickle=True)
            usim = np.load(usim_path, allow_pickle=True)
            # pkg_ori, pkg_re, xml, hash_ori, hash_re
            sim_pairs = [[i[3], i[4], 1] for i in sim]
            sim_name = [[i[0], i[1], i[2], i[2]] for i in sim]
            # pkg1, pkg2, xml1, xml2, hash1, hash2
            usim_pairs = [[i[4], i[5], 0] for i in usim]
            usim_name = [i[:4] for i in usim]
            self.name = []
            self.data = []
            self.data.extend(sim_pairs)
            self.data.extend(usim_pairs)
            self.name.extend(sim_name)
            self.name.extend(usim_name)
            np.savez(npzfile, data=self.data, name=self.name)

        if reshape:
            size = self.data[:, 0].shape
            if len(size) < 3:
                # reshape first two column: hash1, hash2
                self.data[:, 0] = [np.reshape(i, hash_size)
                                   for i in self.data[:, 0]]
                self.data[:, 1] = [np.reshape(i, hash_size)
                                   for i in self.data[:, 1]]

        self.index = [i for i in range(self.__len__())]
        if shuffle_data:
            # shuffle the data by shuffling the indices
            shuffle(self.index)

    def __getitem__(self, idx: int):
        idx = self.index[idx]
        i1, i2, label = self.data[idx]
        if self.transform:
            i1 = self.transform(i1)
            i2 = self.transform(i2)
        return i1, i2, torch.from_numpy(np.array(float(label)))

    def __len__(self):
        return len(self.name)

    def get_info(self, idx: int):
        # return: pkg1, pkg2, xml1, xml2, hash1, hash2
        if np.any(self.name):
            idx = self.index[idx]
            pkg1, pkg2, xml1, xml2 = self.name[idx]
            hash1, hash2, _ = self.data[idx]
            return pkg1, pkg2, xml1, xml2, hash1, hash2


def generate_repackage_dataset(apk_folders: list, list_txt: str,
                               hash_size: Tuple[int, int, int] = (10, 5, 5)):
    a = perf_counter()
    hash_size_str = f"{hash_size[1]}x{hash_size[2]}x{hash_size[0]}"
    g = GenPairsForRepack(apk_folders, list_txt=list_txt,
                          hash_size=hash_size_str)
    sim_pair_count = g.gen_sim_pairs()
    g.gen_unsim_pairs(sim_pair_count)
    LabelledDataSet(hash_size=hash_size)
    b = perf_counter()
    print("successfully generate dataset for repack in", b - a, "s")


def parse_arg_dataset(input_args: list):
    parser = argparse.ArgumentParser(description="Generate dataset "
                                                 "for training and detection")
    parser.add_argument("input_path", help="input paths of repack",
                        type=str, nargs='+', action='append')
    parser.add_argument("app_pair_list", type=str,
                        help="a txt file indicating app "
                             "pairs in a similar app dataset")
    parser.add_argument("--hash_size", "-hs", type=str, default='10,5,5',
                        help="shape of the input UI#")
    _args = parser.parse_args(input_args)
    return _args


if __name__ == "__main__":
    args = parse_arg_dataset(sys.argv[1:])
    try:
        c, t1, t2 = args.hash_size.split(',')
        c, t1, t2 = int(c), int(t1), int(t2)
        generate_repackage_dataset(args.input_path[0],
                                   args.app_pair_list,
                                   hash_size=(c, t1, t2))
    except ValueError:
        print("invalid value for hash_size. example: 10,5,5")
        exit(1)
