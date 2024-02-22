"""Takes a UI hierarchy XML as input, extracts the visible views"""

import xml.dom.minidom as mdom
from xml.dom.minidom import parseString, Element
from xml.parsers import expat
from util.util_xml import xml_init, valid_xml, read_all_nodes
from util.util_log import Logger


class XMLReader:
    """Read a view hierarchy XML (given by uiautomator) and
    turn it into a dictionary

    Attributes:
        err_count (int): The count of xmls that cannot be parsed
            successfully (due to a xml file in invalid format)
        nodes (list): The extracted xml node objects
        node_dicts (list): A dict holds info of each xml nodes.
        naive_xml (bool): False when use uiautomator2 xml, if the hierarchy
            is dumped by naive adb, then true
    """
    def __init__(self, xml_path: str, only_visible: bool = True,
                 naive_xml: bool = False):
        self._logger = Logger()
        try:
            self.err_count: int = 0
            self.nodes = list()
            self._only_visible = only_visible
            xml = valid_xml(xml_path)
            if xml is None:
                self._logger.get_logger.warn(f"empty xml data in {xml_path}")
            self._root = parseString(xml)
            xml_init(self._root)
            if not self._only_visible:
                read_all_nodes(self.nodes, self._root, True)
            else:
                self.read_all_visible_nodes(self.nodes, self._root, True, naive_xml)
            self.node_dicts = [self.get_dict(i) for i in self.nodes]
            self.node_dicts = [n for n in self.node_dicts if 'lt' in n]
            self._logger.get_logger.debug(f"successfully extract nodes from {xml_path}")
        except expat.ExpatError as e:
            self.err_count += 1
            self._logger.get_logger.warn(f"invalid xml {xml_path}: {e}")

    def get_dict(self, node: mdom.Element) -> dict:
        _dict = dict()
        _dict['name'] = node.getAttribute('class')
        bounds = node.getAttribute('bounds').replace(']', '').split('[')
        if len(bounds) > 1:
            _dict['lt'], _dict['rb'] = bounds[1].split(','), bounds[2].split(',')
        if not self._only_visible:
            _dict['visible'] = node.getAttribute('visible-to-user')
        _dict['text'] = node.getAttribute('text')
        _dict['interact'] = node.getAttribute('clickable').startswith('t') \
            or node.getAttribute('long-clickable').startswith('t') \
            or node.getAttribute('checkable').startswith('t')
        return _dict

    @classmethod
    def read_all_visible_nodes(cls, node_list: list, node: Element,
                               is_root: bool = True, naive_xml: bool = False):
        """Read all **visible** nodes in the xml.
        Here, the following nodes are considered as invisible:

            1. LayoutGroups
            2. Scrollers
            3. Reported as not visible from uiautomator
            4. TextViews without any text

        Args:
            node_list (list): The list to store the nodes. please make it empty
            node (Element): The root node of a xml object
            is_root (bool): Just make it the default value
            naive_xml (bool): False when use uiautomator2 xml, if the hierarchy
              is dumped by naive adb, then true
        """
        if not is_root:
            _class = node.getAttribute('class')
            if not _class.endswith('Layout'):
                if not _class.split('.')[-1].startswith('Scroll'):
                    if not _class.endswith('Group'):
                        if not naive_xml:
                            if node.getAttribute('visible-to-user').startswith('t'):
                                if not _class.endswith('TextView'):
                                    node_list.append(node)
                                else:
                                    text = node.getAttribute('text')
                                    if len(text.strip()) > 0:
                                        node_list.append(node)
                        else:
                            if not _class.endswith('TextView'):
                                node_list.append(node)
                            else:
                                text = node.getAttribute('text')
                                if len(text.strip()) > 0:
                                    node_list.append(node)
        if node.hasChildNodes():
            for child in node.childNodes:
                cls.read_all_visible_nodes(node_list, child, False, naive_xml=naive_xml)
