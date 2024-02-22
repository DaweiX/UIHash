"""Analyse an app and extract its features"""

import os

from androguard.core.bytecodes.apk import APK
from os.path import basename, exists, join

from androguard.core.bytecodes.axml import AXMLPrinter

from util.util_platform import init_path, get_logger
from os import mkdir, makedirs
from asn1crypto.x509 import Certificate
from json import dumps
import json
from typing import Any


class Meta:
    """meta info extracted from an apk"""

    def __init__(self, output_root_path: str) -> None:
        self.__meta = {}
        if not exists(output_root_path):
            makedirs(output_root_path)
        self.output_root_path = output_root_path

    def get_meta(self) -> dict:
        return self.__meta

    def set_meta(self, key: str, value: object) -> None:
        self.__meta[key] = value

    def get_meta_item(self, key) -> Any:
        if key not in self.__meta:
            return None
        return self.__meta[key]

    def write_meta(self) -> str:
        try:
            with open(join(self.output_root_path, "meta.json"),
                      encoding="utf-8", mode="w+") as f:
                json_str = json.dumps(self.__meta)
                f.write(json_str)
        except Exception as e:
            return "err: " + str(e)
        return "ok"


class ExtractApk:
    def __init__(self, apk_file: str, sha256: str,
                 out_dir_root: str, logging_level: str) -> None:
        self.logger = get_logger(self.__class__.__name__, logging_level)
        self.logger.debug(f"meta start. output dir: {out_dir_root}/{sha256}")
        out_dir_root = init_path(out_dir_root)
        file_name = basename(apk_file)
        self.out_dir_apk = join(out_dir_root, sha256)
        self.__meta = Meta(self.out_dir_apk)
        self.__meta.set_meta("file_name", file_name)
        self.__meta.set_meta("file_sha256", sha256)
        self.__apk = APK(apk_file)

    def summary(self):
        """apk info extracted from manifest"""
        self.get_package()
        self.get_apk_version()
        self.get_app_name()
        self.is_signed()

        # four android app components
        self.get_providers()
        self.get_receivers()
        self.get_services()
        self.get_activities()

        self.get_permissions()
        self.get_libraries()
        self.__meta.write_meta()

    def get_permissions(self) -> None:
        permissions = self.__apk.get_permissions()
        p1 = self.__apk.get_requested_aosp_permissions()
        p2 = self.__apk.get_requested_third_party_permissions()

        permissions.extend(p1)
        permissions.extend(p2)

        permissions = list(set(permissions))
        self.__meta.set_meta("permissions", permissions)

    def get_activities(self) -> None:
        activities = self.__apk.get_activities()
        self.__meta.set_meta("activities", activities)

    def get_package(self) -> None:
        package = self.__apk.get_package()
        self.__meta.set_meta("package", package)

    def get_app_name(self) -> None:
        app_name = self.__apk.get_app_name()
        self.__meta.set_meta("app name", app_name)

    def print_certs(self) -> None:
        cert_list = self.__apk.get_certificates()
        cert_dir = join(self.out_dir_apk, "certs")
        cert_list = iter(cert_list)
        if not exists(cert_dir):
            mkdir(cert_dir)
        i = 0
        while True:
            try:
                cert: Certificate = cert_list.__next__()
                with open(join(cert_dir, f"{i}.txt"), "w+") as f:
                    f.writelines([f"sha256: {cert.sha256_fingerprint}\n",
                                  f"issuer: {cert.issuer.human_friendly}\n",
                                  f"subject: {cert.subject.human_friendly}\n",
                                  f"hash algo: {cert.hash_algo}\n",
                                  f"signature algo: {cert.signature_algo}\n",
                                  f"serial: {cert.serial_number}\n",
                                  f"contents: {str(cert.contents)}"])
                    i += 1
            except StopIteration:
                break

    def print_manifest_xml(self) -> str:
        try:
            xml = self.__apk.get_android_manifest_axml().get_xml()
            with open(join(self.out_dir_apk, "manifest.xml"),
                      encoding="utf-8", mode="w+") as f:
                f.write(str(xml, encoding="utf-8"))
        except Exception as e:
            self.logger.warning("err: " + str(e))
            return "err: " + str(e)
        return "ok"

    def get_file_crcs(self) -> str:
        files = self.__apk.get_files_crc32()
        try:
            with open(join(self.out_dir_apk, "files.json"),
                      encoding="utf-8", mode="w+") as f:
                f.write(dumps(files))
        except Exception as e:
            self.logger.warning("err: " + str(e))
            return "err: " + str(e)
        return "ok"

    def get_libraries(self) -> None:
        libraries = self.__apk.get_libraries()
        self.__meta.set_meta("libs", libraries)

    def get_providers(self) -> None:
        providers = self.__apk.get_providers()
        self.__meta.set_meta("providers", providers)

    def get_receivers(self) -> None:
        receivers = self.__apk.get_receivers()
        self.__meta.set_meta("receivers", receivers)

    def get_services(self) -> None:
        services = self.__apk.get_services()
        self.__meta.set_meta("services", services)

    def is_signed(self) -> None:
        is_signed = self.__apk.is_signed()
        self.__meta.set_meta("signed", is_signed)

    def get_apk_version(self) -> None:
        version = self.__apk.get_androidversion_name()
        self.__meta.set_meta("version", version)

    def extract_icon(self) -> str:
        icon_path = self.__apk.get_app_icon()
        icon_name = icon_path.split(os.sep)[-1]
        img = self.__apk.get_file(icon_path)
        if not len(img):
            return "empty"
        try:
            with open(f"{self.out_dir_apk}{os.sep}"
                      f"icon_{icon_name}", mode="bw+") as f:
                f.write(img)
        except Exception as e:
            self.logger.warning("err: " + str(e))
            return "err: " + str(e)
        return "ok"

    def print_layout_files(self) -> str:
        files = self.__apk.get_files()
        # we only consider default layout files here
        layout_files = [f for f in files if
                        f.startswith(f"res{os.sep}layout{os.sep}")]
        layout_out_path = join(self.out_dir_apk, "layout")
        if not exists(layout_out_path):
            mkdir(layout_out_path)
        try:
            if len(layout_files) == 0:
                return "empty_res_layout"
            for layout_file in layout_files:
                xml = AXMLPrinter(self.__apk.get_file(layout_file)).get_xml()
                with open(join(layout_out_path, layout_file.split(os.sep)[-1]),
                          encoding="utf-8", mode="w+") as f:
                    f.write(str(xml, encoding="utf-8"))
        except Exception as e:
            self.logger.warning("err: " + str(e))
            return "err: " + str(e)
        return "ok"

    def extract_dex_files(self) -> str:
        files = self.__apk.get_files()
        dex_files = [f for f in files if f.endswith(".dex")]
        dex_out_path = join(self.out_dir_apk, "dex")
        if not exists(dex_out_path):
            mkdir(dex_out_path)
            if len(dex_files) == 0:
                return "empty_dex_layout"
        try:
            for dex_file in dex_files:
                dex_bytes = self.__apk.get_file(dex_file)
                with open(join(dex_out_path,
                               dex_file.split(os.sep)[-1]), mode="wb+") as f:
                    f.write(dex_bytes)
        except Exception as e:
            self.logger.warning("err: " + str(e))
            return "err: " + str(e)

        return "ok"
