"""Help functions for reading or operating files"""

from hashlib import sha256
from os import walk
from os.path import join


def list_apks(root_path: str) -> list:
    """List all the apk files in a folder

    Args:
        root_path (str): The root input path. This 
          function will also read apks from the subfolders
    Returns:
        A list including all the complete pathes of 
          apk files found
    """
    all_apk = []
    for dirpath, _, filenames in walk(root_path):
        for name in filenames:
            if name.endswith('.apk'):
                all_apk.append(join(dirpath, name))
    return all_apk


def cal_sha256(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        hashobj = sha256()
        hashobj.update(f.read())
        return hashobj.hexdigest()
