"""Given XML nodes, generate UI#"""

from typing import Tuple
import PIL
import numpy as np
from util.util_math import get_iou, amp_small_scaler
from PIL import Image
from os.path import join
from xml2nodes import XMLReader


class Nodes2Hash:
    """ Takes view dicts as input and output UIHash

    Args:
        channels (int): the expected channel number in UIHash
        h_v_ticks ((int, int)): the grid size for each channel

    Attributes:
        channels (int): UIHash channel number
    """
    def __init__(self, h_v_ticks: Tuple[int, int], channels: int):
        self._screen_h, self._screen_v = 0, 0
        self._h_tick = h_v_ticks[0]
        self._v_tick = h_v_ticks[1]
        self.channels = channels

    @staticmethod
    def fine_tune_grid_lt(num: float, tick: int, size: float,
                          t: float = 0.25) -> int:
        """ Adjust view's start grid by its left/top coords, like:
          * 0.1 -> 0, 0.9 -> 1
          * 1.1 -> 1, 1.9 -> 2
          * 2.1 -> 2, 2.9 -> 2

        Args:
            num (float): input value
            tick (int): `self._h_tick` or `self._v_tick`
            size (float): view width or height (from 0 to self._?_tick)
            t (float): threshold to determine when a view occupies a grid
                when a part of it is in the grid
        """
        diff = num - int(num)
        if diff < 0.5:
            return int(num)
        try:
            if (1 - diff) / tick / size < t:
                # if only a tiny part located in the left/top corner
                # then just go right/down
                return min(int(num) + 1, tick - 1)
        except ZeroDivisionError:
            return int(num)
        else:
            # hold in this grid
            return int(num)

    @staticmethod
    def fine_tune_grid_rb(num: float, tick: int, size: float,
                          t: float = 0.25) -> int:
        """ Adjust view's start grid by its right/bottom coords

        Args:
            num (float): input value
            tick (int): `self._h_tick` or `self._v_tick`
            size (float): view width or height (from 0 to self._?_tick)
            t (float): threshold to determine when a view occupies a grid
                when a part of it is in the grid
        """
        diff = num - int(num)
        if diff >= 0.5:
            return int(num)
        try:
            if diff / tick / size < t:
                # go left/up
                return max(int(num) - 1, 0)
        except ZeroDivisionError:
            return int(num)
        else:
            # keep in this grid
            return int(num)

    def assign_hash_grid(self, nodes: list):
        """ Put views into the grids where they should be.
        For each view (dict), keys `area` and `grids`
        will be appended. Besides, if the view is already
        out-of-screen, then this function will skip the view:
        the key `grids` will not be added, and the `area` will
        be zero. As a parallel way for UIHash generationg, this
        function is not used by default in the release.

        Args:
            nodes (list): input node dict list
        """
        for n in nodes:
            n['area'] = 0
            h, v = self._screen_h, self._screen_v

            # if the left top corner out of the
            # right/bottom bounds of screen, continue
            h_1 = int(n['lt'][0]) / (h / self._h_tick)
            v_1 = int(n['lt'][1]) / (v / self._v_tick)
            if h_1 >= self._h_tick or v_1 >= self._v_tick:
                continue

            h_2 = int(n['rb'][0]) / (h / self._h_tick)
            v_2 = int(n['rb'][1]) / (v / self._v_tick)
            if h_2 < 0 or v_2 < 0:
                continue

            width = int(n['rb'][0]) - int(n['lt'][0])
            height = int(n['rb'][1]) - int(n['lt'][1])
            width = float(width) / h
            height = float(height) / v

            h_1 = self.fine_tune_grid_lt(h_1, self._h_tick, width)
            v_1 = self.fine_tune_grid_lt(v_1, self._v_tick, height)
            h_2 = self.fine_tune_grid_rb(h_2, self._h_tick, width)
            v_2 = self.fine_tune_grid_rb(v_2, self._v_tick, height)

            n['grids'] = (h_1, v_1, h_2, v_2)
            n['area'] = width * height

    def gen_uihash(self, xml_path: str, nodes: list = None,
                   naive_xml: bool = False) -> np.array:
        """ Given a hierarchy xml, generate uihash. If the view is
        already out-of-screen, then the view will be skipped.
        The value in a grid will be originally given by the IoU
        between the view and the grid, then the value will be
        amplified by a log function. We determine which channel to
        put the view firstly by its reidentified view type. If the
        reidentification model is not very sure of its decision,
        then we compromise to the claimed view type.

        Args:
            nodes (list): view nodes in ui. if not provided, parse xml
              nodes here
            xml_path (str): the input hierarchy file.
            naive_xml (bool): false when use uiautomator2 xml, if the hierarchy
              is dumped by naive adb, then true
        """
        if nodes is None:
            nodes = XMLReader(xml_path, naive_xml=naive_xml).node_dicts
        try:
            img = Image.open(f"{xml_path[:-4]}.jpg")
        except PIL.UnidentifiedImageError:
            print(f"unable to load image {xml_path[:-4]}.jpg")
            return None
        except FileNotFoundError:
            print(f"image {xml_path[:-4]}.jpg not exists")
            return None
        self._screen_h = img.size[0]
        self._screen_v = img.size[1]
        for n in nodes:
            h, v = self._screen_h, self._screen_v
            # if the left top corner out of the
            # right/bottom bounds of screen, continue
            h_min, v_min = int(n["lt"][0]), int(n["lt"][1])
            if h_min >= h or v_min >= v:
                continue
            h_max, v_max = int(n["rb"][0]), int(n["rb"][1])
            if h_max < 0 or v_max < 0:
                continue
            n["area4grids"] = [0] * (self._v_tick * self._h_tick)

            for i in range(self._v_tick * self._h_tick):
                # calculate IoU (Intersection over Union) for each grid
                h_start_idx, v_start_idx = i % self._h_tick, \
                                           int(i / self._h_tick)
                size_unit_h = float(h) / self._h_tick
                size_unit_v = float(v) / self._v_tick
                h1 = h_start_idx * size_unit_h
                v1 = v_start_idx * size_unit_v
                h2, v2 = h1 + size_unit_h, v1 + size_unit_v
                iou = get_iou((h_min, v_min, h_max, v_max),
                              (h1, v1, h2, v2))
                width = int(n["rb"][0]) - int(n["lt"][0])
                height = int(n["rb"][1]) - int(n["lt"][1])
                width = width / (float(h) / self._h_tick)
                height = height / (float(v) / self._v_tick)
                area = width * height  # n area (compared to one grid)
                if area == 0:
                    continue
                # when the iou is only a tiny part of the view
                # we dont consider it
                t = -1
                if self._h_tick == self._v_tick == 5:
                    t = 0.07
                elif self._h_tick == 4 and self._v_tick == 3:
                    t = 0.12
                elif self._h_tick == 2 and self._v_tick == 2:
                    t = 0.2
                if iou / area > t:
                    n['area4grids'][i] = iou

        mat = np.zeros((self.channels, self._h_tick * self._v_tick))
        type_dict = dict()
        type_file = join(xml_path[:-4], "classify.txt")
        with open(type_file, mode='r') as f:
            raw_type_dict = eval(f.readline())
        for key in raw_type_dict:
            index, original_type = key.split('_', 1)
            reidentified_type = raw_type_dict[key]
            type_dict[index] = (original_type, reidentified_type)
        for i, n in enumerate(nodes):
            if "area4grids" not in n:
                continue
            # get type of the node
            ori_type = n["name"]
            _c = -1
            if str(i) in type_dict:
                ori_type, _c = type_dict[str(i)]
            else:
                continue
            if _c < 0:
                # use the declared class name instead
                if "RadioB" in ori_type:
                    _c = 1
                elif "ToggleB" in ori_type:
                    _c = 6
                elif "Button" in ori_type:
                    _c = 0
                elif "Check" in ori_type:
                    _c = 1
                elif "ListView" in ori_type:
                    _c = 3
                elif "TextView" in ori_type:
                    _c = 5
                elif "EditT" in ori_type:
                    _c = 2
                elif "Switch" in ori_type:
                    _c = 6
                elif "CompoundButton" in ori_type:
                    _c = 1
                elif "TabW" in ori_type or "$Tab" in ori_type:
                    _c = 4
                elif "Spinner" in ori_type:
                    _c = 7
                elif "Bar" in ori_type:
                    _c = 7
                else:
                    _c = 7

            for j, k in enumerate(n['area4grids']):
                mat[_c][j] += k
            # update the values
            mat[_c] = [amp_small_scaler(i) for i in mat[_c]]
        return mat
