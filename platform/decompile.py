"""Decompile an app for code-level inspection"""

from subprocess import check_call, CalledProcessError
from os.path import exists, join, abspath
from os import makedirs, mkdir
from time import perf_counter

from util.util_platform import get_logger


class DexDecompiler:
    def __init__(self,
                 apk_file: str,
                 out_dir_root: str,
                 sha256: str,
                 logging_level: str):
        self.logger = get_logger(self.__class__.__name__, logging_level)
        self.logger.debug(f"decompile start. output dir: {out_dir_root}/{sha256}")
        output_root_path = join(out_dir_root, sha256, "java")
        if not exists(output_root_path):
            makedirs(output_root_path)

        self.apk_path = apk_file
        self.output_root_path = output_root_path

    def run_jadx(self, threads: int = 8) -> str:
        _bin = abspath(join("tools", "jadx", "bin", "jadx.bat"))
        if not exists(_bin):
            error = f"jadx bin not found in: {_bin}"
            self.logger.error(error)
            return error
        opt_path = join(self.output_root_path, "jadx")
        if not exists(opt_path):
            mkdir(opt_path)
        try:
            t1 = perf_counter()
            check_call([_bin, abspath(self.apk_path), "--no-res", "-ds", abspath(opt_path),
                        "-j", str(threads), "--no-debug-info", "--cfg"],)
            t2 = perf_counter()
            self.logger.debug(f"jadx finished in {t2 - t1} s")
            return "ok"
        except CalledProcessError as e:
            self.logger.warning(f"error when running jadx: {e}")
            return f"err: {e}"

    def run_dex2jar(self):
        raise NotImplementedError
