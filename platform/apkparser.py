"""Main program to extract various app features from an apk file"""

import argparse
import json
from time import perf_counter
from zipfile import BadZipFile

from os import listdir, makedirs
from os.path import join, exists
import sys
import _locale
import os

curpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.split(curpath)[0]
if rootpath not in sys.path:
    sys.path.append(rootpath)

from extract_apk import ExtractApk
from util.util_platform import get_logger, check_ok
from util.util_file import cal_sha256
from decompile import DexDecompiler

_locale._getdefaultlocale = (lambda *aargs: ["en", "utf-8"])

PARSED_LIST = "parsed_list.txt"
STATUS_FILE = "status.txt"
HASHES_LIST = "hashes.txt"


class APKParser:
    def __init__(self, _args):
        self.apks_root = _args.apk_root_path
        self.output_root = _args.out_root_path
        self.level = _args.logging
        self.overwrite = _args.overwrite
        self.manifest = _args.manifest
        self.icon = _args.icon
        self.cert = _args.cert
        self.layout = _args.layout
        self.file = _args.file
        self.apk_list = [i for i in listdir(self.apks_root) if i.endswith(".apk")]
        self.logger = get_logger(self.__class__.__name__, self.level)
        self.exist_status = dict()
        self.java_dex2jar = False
        self.java_jadx = False
        if isinstance(_args.java, list):
            if "dex2jar" in _args.java:
                self.java_dex2jar = True
            if "jadx" in _args.java:
                self.java_jadx = True

        if self.overwrite:
            self.logger.warning("overwrite mode is on")

        if not exists(self.output_root):
            makedirs(self.output_root)
        # list all apks and know which ones
        # are already parsed (partly or completely)
        parsed_list_file = join(self.output_root, PARSED_LIST)
        if not exists(parsed_list_file):
            self.parsed_hashes = list()
        else:
            with open(parsed_list_file, mode="r") as f:
                self.parsed_hashes = [s[:-1] for s in f.readlines()]

        self.logger.info("fetching hashes for all apks...")
        hash_list_file = join(self.output_root, HASHES_LIST)
        self.todo_hashes = []
        if exists(hash_list_file):
            with open(hash_list_file, mode="r", encoding="utf-8") as f:
                for line in f:
                    self.todo_hashes.append(line.split(' ')[0])
            self.logger.info("hash list loaded")
        else:
            with open(hash_list_file, mode="w+", encoding="utf-8") as f:
                for a in self.apk_list:
                    h = cal_sha256(join(self.apks_root, a))
                    self.todo_hashes.append(h)
                    f.write(f"{h} {a}\n")
                    self.logger.debug(h)
        self.new_hashes = list()

    def should_fetch(self, item: str) -> bool:
        if item not in self.__dict__:
            # the function is not required
            return False
        if not self.__getattribute__(item):
            return False
        if check_ok(self.exist_status, item) and not self.overwrite:
            self.logger.debug(
                f"already parse {item} successfully, skip")
            return False
        # in overwrite mode, refetch even if there is already valid data
        self.logger.debug(
            f"\thandling {item}")
        return True

    def run_meta(self, apk_name: str, apk_hash: str,
                 exist_status: dict) -> dict:
        t1 = perf_counter()
        meta_parser = ExtractApk(join(self.apks_root, apk_name),
                                 apk_hash, self.output_root, self.level)
        if self.should_fetch("manifest"):
            meta_parser.summary()
            exist_status["manifest"] = meta_parser.print_manifest_xml()
        if self.should_fetch("icon"):
            exist_status["icon"] = meta_parser.extract_icon()
        if self.should_fetch("cert"):
            meta_parser.print_certs()
            exist_status["cert"] = "ok"
        if self.should_fetch("file"):
            exist_status["file"] = meta_parser.get_file_crcs()
        if self.should_fetch("layout"):
            exist_status["layout"] = meta_parser.print_layout_files()
        if self.should_fetch("dex"):
            exist_status["dex"] = meta_parser.extract_dex_files()
        t2 = perf_counter()
        self.logger.debug(f"apk components parsed done in {t2 - t1} s")
        return exist_status

    def run_decompile(self, apk_name: str, apk_hash: str,
                      exist_status: dict) -> dict:

        if self.java_jadx or self.java_dex2jar:
            code_decompiler = DexDecompiler(apk_file=join(self.apks_root, apk_name),
                                            sha256=apk_hash,
                                            out_dir_root=self.output_root,
                                            logging_level=self.level)
            if self.should_fetch("java_jadx"):
                exist_status["java_jadx"] = code_decompiler.run_jadx()
            if self.should_fetch("dex2jar"):
                exist_status["dex2jar"] = code_decompiler.run_dex2jar()
        return exist_status

    def run(self):
        for i, apk in enumerate(self.apk_list):
            self.logger.info(f"({i + 1}/{len(self.apk_list)}) {apk}")
            current_hash = self.todo_hashes[i]
            # load parse status
            status_file = join(self.output_root, current_hash, STATUS_FILE)
            if exists(status_file):
                with open(status_file, mode="r") as f:
                    self.exist_status = json.loads(f.read())
            else:
                self.exist_status = {}
            # parse the apk
            try:
                current_status = self.run_meta(apk, current_hash, self.exist_status)
            except BadZipFile:
                self.logger.warning(f"{apk} is a bad zipfile")
                continue

            # decompile and get java
            current_status = self.run_decompile(apk, current_hash, current_status)

            # mark current file as handled (partly or completely)
            with open(status_file, mode="w+") as f:
                f.write(json.dumps(current_status))
            if current_hash not in self.parsed_hashes:
                self.new_hashes.append(current_hash + '\n')

        if len(self.new_hashes) > 0:
            with open(join(self.output_root, PARSED_LIST), mode="a+") as f:
                f.writelines(self.new_hashes)


def parse_arg(input_args: list):
    parser = argparse.ArgumentParser(
        description="Extract various features from android packages")
    parser.add_argument("apk_root_path",
                        help="the path where you put your apks")
    parser.add_argument("out_root_path",
                        help="the path where you find the outputs")
    parser.add_argument("--logging",
                        help="logging level, default: info",
                        default="debug",
                        choices=["debug", "info", "warn"])
    parser.add_argument("--overwrite", "-o", action="store_true",
                        help="if assigned, then work in overwrite mode, "
                             "otherwise skip existing items (default)")
    parser.add_argument("--manifest", "-m", action="store_true",
                        help="extract app manifest")
    parser.add_argument("--icon", "-i", action="store_true",
                        help="extract app icon")
    parser.add_argument("--cert", "-c", action="store_true",
                        help="extract app certificate")
    parser.add_argument("--layout", "-l", action="store_true",
                        help="extract app layout files")
    parser.add_argument("--file", "-f", action="store_true",
                        help="dump file dictionary and file hashes of the apk")
    parser.add_argument("--java", "-j", nargs="+",
                        choices=["jadx", "dex2jar"],
                        help="decompile java and get cfg using certain tools")
    _args = parser.parse_args(input_args)
    return _args


if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])
    APKParser(args).run()
