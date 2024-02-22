"""Generate UI# for a large scale UIs"""

import os
from os import listdir, makedirs
from typing import Tuple

import numpy as np
from os.path import exists, join
from time import perf_counter
import sys
import argparse

curpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.split(curpath)[0]
if rootpath not in sys.path:
    sys.path.append(rootpath)

from xml2nodes import XMLReader
from nodes2hash import Nodes2Hash


def gen_hash_data(ipt_paths: list,
                  opt_path: str,
                  view_img_dataset: str,
                  hash_grid_size: Tuple[int, int] = (5, 5),
                  filter_few_nodes: int = 6,
                  input_dataset_name: str = "",
                  naive_xml: bool = False):
    """ Generate uihash for one or more input path(s). Each input path
    includes screenshots, hierarchies, and re-identified view types
    for each UI

    Args:
      filter_few_nodes (int): 0 to remove filter, otherwise the threshold
        of the minimal accepted visible nodes in a UI
      ipt_paths (list): The complete input paths, each one represent a dataset
        or a subset (e.g., original and repackage apps in RePack). Each folder
        in the path holds the raw UI data (screenshot, view hierarchy, and images
        and types of each visible view)
      opt_path (str): The output path to store the uihash results
      view_img_dataset (str): The path of the training data for view type
        reidentification. The type total num, i.e., channel num for uihash, is
        determined according to this
      input_dataset_name (str): A magic paremeter, please make it 'ori' when
        the only ipt_path is the original apps in a labeled dataset like RePack,
        and 're' for the repackaged apps. Just keep it unset when working on an
        unlabeled dataset
      hash_grid_size ((int, int)): The (2D) grid size expected for uihash
      naive_xml (bool): false when use uiautomator2 xml, if the hierarchy
        is dumped by naive adb, then true
    """

    if len(opt_path) == 0:
        opt_path = join(os.path.abspath(os.path.dirname(__file__)),
                        "..", "output", "hash")

    if not exists(opt_path):
        makedirs(opt_path)
        print("hash npy output path:", opt_path)

    hash_list = list()
    apk_xml_list = list()
    classes_names = listdir(view_img_dataset)
    classes_names = [c for c in classes_names if
                     os.path.isdir(os.path.join(view_img_dataset, c))]
    k = 0
    # +1: others (images in most cases)
    type_number = len(classes_names) + 1
    for folder in ipt_paths:
        print(f"----------entering: {folder}----------")
        pkgs = listdir(folder)
        total = len(pkgs)
        hasher = Nodes2Hash(hash_grid_size, type_number)

        for i, pkg in enumerate(sorted(pkgs)):
            print(f'{i + 1}/{total} {pkg}')
            xmls = listdir(join(folder, pkg))
            xmls = [i for i in xmls if i.endswith("xml")]
            for xml in xmls:
                xml_path = join(folder, pkg, xml)
                nodes = XMLReader(xml_path, naive_xml=naive_xml).node_dicts
                if filter_few_nodes > 0:
                    if len(nodes) < filter_few_nodes:
                        continue
                ui_hash = hasher.gen_uihash(xml_path=xml_path, nodes=nodes)
                if ui_hash is not None:
                    hash_list.append(ui_hash)
                    apk_xml_list.append(f"{pkg} {xml}")
                    k += 1
    print(k, "xmls to hashes.")

    postfix = f"_{input_dataset_name}" \
              f"_{hash_grid_size[0]}x{hash_grid_size[1]}" \
              f"x{type_number}"
    postfix = postfix.replace("__", "_")
    np.save(join(opt_path, f"hash{postfix}.npy"),
            hash_list, allow_pickle=True)
    np.save(join(opt_path, f"name{postfix}.npy"),
            apk_xml_list, allow_pickle=True)


def parse_arg_uihash(input_args: list):
    parser = argparse.ArgumentParser(description="Turns UI into UI#")
    parser.add_argument("input_path", help="input paths",
                        type=str, nargs='+', action='append')
    parser.add_argument("view_image_path", type=str,
                        help="path for the view image dataset")
    parser.add_argument("--output_path", "-o", help="output path",
                        type=str, default="")
    parser.add_argument("--naivexml", "-n", action="store_true",
                        help="assign it when when use naive adb, "
                             "and ignore it when use uiautomator2 xml")
    parser.add_argument("--dataset_name", "-d", default="", type=str,
                        help="make it 'ori' when the only ipt_path is the original apps "
                             "in a labeled dataset like RePack, and 're' for the repackaged "
                             "apps. Just keep it unset when working on an unlabeled dataset")
    parser.add_argument("--grid_size", "-g", default='5,5', type=str,
                        help="expected grid size for UI#. format: tick_horizontal,tick_vertical")
    parser.add_argument("--filter", "-f", default=5, type=int,
                        help="0 to remove filter, otherwise the threshold of "
                             "the minimal accepted visible nodes in a UI")

    _args = parser.parse_args(input_args)
    return _args


if __name__ == "__main__":
    args = parse_arg_uihash(sys.argv[1:])
    try:
        t1, t2 = args.grid_size.split(',')
        t1, t2 = int(t1), int(t2)
        grid_size = (t1, t2)
        start = perf_counter()
        gen_hash_data(ipt_paths=args.input_path[0],
                      opt_path=args.output_path,
                      view_img_dataset=args.view_image_path,
                      hash_grid_size=grid_size,
                      filter_few_nodes=args.filter,
                      input_dataset_name=args.dataset_name,
                      naive_xml=args.naivexml)
        end = perf_counter()
        print(f"time cost {args.grid_size}:", end - start)

    except ValueError:
        print("invalid value for grid_size. example: 5,5")
        exit(1)
