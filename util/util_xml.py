"""Help functions for reading and parsing XML documents"""

from os.path import isfile
from typing import Tuple
from xml.dom.minidom import Document, parseString, Element
from zipfile import BadZipFile, ZipFile
import re
from androguard.core.bytecodes.axml import AXMLPrinter


def dump_activity_from_apk(apk_path: str):
    """ Fetch package name and activity list from apk manifest

    Args:
        apk_path (str): Path of an apk file

    Returns:
        Package name (@1, str) and activities
          claimed in the manifest (@2, list)

    Raises:
        BadZipFile: In this case, the return
          value will be ('badfile', None)
    """
    if not isfile(apk_path):
        print(f'File {apk_path} not exist')
        return None

    _file = "AndroidManifest.xml"
    try:
        with ZipFile(apk_path, 'r') as a:
            text = AXMLPrinter(a.read(_file))
            xml = text.get_xml()
            dom = parseString(xml)
            alist = dom.getElementsByTagName("activity")
            package_name = dom.getElementsByTagName(
                'manifest')[0].getAttribute('package')
            return package_name, [
                a.getAttribute('android:name') for a in alist]
    except BadZipFile:
        return 'badfile', None


def is_focus(ipt: str) -> bool:
    """If needed, assign some keywords for UI searching.
    Note that this function should be only used for
    sorting interesting UIs to the top instead of filtering, 
    because an adversary can make the activity names obscure

    Args:
        ipt (str): An activity name

    Returns:
        A bool, indicating whether the activity is
          potentially security-related or privacy-related
    """
    ipt = ipt.lower()
    focus_list = ['login', 'logon', 'home']
    for i in focus_list:
        if i in ipt:
            return True
    return False


def valid_xml(xml_path: str) -> str or None:
    """Make the xml in the valid format

    Args:
        xml_path (str): Path to the xml file

    Returns:
        A xml string or None
    """
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml = f.read()
    if not xml:
        return None
    if xml.count('<?xml') == 1:
        return xml
    elif xml.count('<?xml') > 1:
        a = re.findall(r'(?<=\?>)[\s\S]+?(?=<\?)|(?<=\?>)[\s\S]+?$', xml)
        xml_tag = re.search(r'<\?.+?\?>', xml).group()
        lengths = [len(i) for i in a]
        new_xml = xml_tag + a[lengths.index(max(lengths))]
        return new_xml
    else:
        return None


def is_removal(node: Element,
               keywords: str = 'com.microvirt') -> bool:
    """Help to remove android sys nodes (like the top banner)
    and views added by other sys apps (i.e., float items)

    Args:
        node (Element): A xml node
        keywords (str): Split by comma, indicating all the keywords that
          are not related to UI analysis (e.g., global float area, or
          some controls drawed by certain simulator/OS)

    Returns:
        whether a n is of no sense
    """
    package = node.getAttribute('package')
    if package == 'android':
        return True
    if package.count('com.android.systemui'):
        return True
    counts = sum([package.count(a) for a in keywords.split(',')])
    if counts > 0:
        return True
    return False


def xml_init(node: Document):
    """Initialize a xml doc by removing its
    comment nodes or text nodes

    Args:
        node (Document): The xml doc object
    """
    if node.childNodes:
        for child in node.childNodes[:]:
            if child.nodeType == child.TEXT_NODE or \
                    child.nodeType == child.COMMENT_NODE:
                node.removeChild(child)
                continue
            if is_removal(child):
                node.removeChild(child)
                continue
            xml_init(child)


def remove_sysnode(xml: str) -> Tuple[bool, Element]:
    """Initialize a hierarchy and remove its root node

    Args:
        xml (str): Input hierarchy string

    Returns:
        The first return value is a bool, indicating whether
          the hierarchy is empty. The second return value is
          the output hierarchy dom
    """
    dom = parseString(xml)
    xml_init(dom)
    return dom.getElementsByTagName('hierarchy')[0].hasChildNodes(), dom


def read_all_nodes(node_list: list, node: Element, isroot: bool = True):
    """Read xml nodes from a root node
    
    Args:
        node_list (list): An input list for xml nodes, usually an empty one.
          The results will be saved in the list.
        node (Element): The root xml node
        isroot (bool): Just make it True when calling the
          function somewhere else
    """
    if not isroot:
        node_list.append(node)
    if node.hasChildNodes():
        for child in node.childNodes:
            read_all_nodes(node_list, child, False)
