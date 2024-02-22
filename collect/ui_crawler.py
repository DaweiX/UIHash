"""Collect UIs dynamically at runtime"""

import argparse
import os.path
import subprocess
import time
from subprocess import check_output
from time import perf_counter, sleep
from os.path import exists, join
from os import mkdir, listdir
import sys

curpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.split(curpath)[0]
if rootpath not in sys.path:
    sys.path.append(rootpath)

from util.util_log import Logger
from util.util_xml import is_focus, remove_sysnode, \
    dump_activity_from_apk, read_all_nodes
from util.util_file import cal_sha256
from device import Device


class UICrawler:
    """Given a device, traverse apks on it and extract UIs at runtime.
    The parameters of this class are read from the command instead of
    a config file, because in practice we can launch multiple crawler
    in parallel (with different devices and apk ranges) to accelerate
    the dynamic data collecting.

    Attributes:
        device (Device): The device object we interact (via adb) with
    """

    def __init__(self, input_args):
        # after 10 successive failure, we turns to the next app
        self._FAIL_TRY_TIMES = 10
        self._ip: str = input_args.ip
        self._apk_folder: str = input_args.apk_folder
        # we record the last position of the apk list to continue from
        # it at the next run
        self._resume = input_args.resume
        file_path = os.path.dirname(self._apk_folder)
        file_name = f'index_{self._ip.replace(":", "-")}' \
                    f'-{self._apk_folder.split(os.sep)[-1]}.txt'
        self._LAST_POS_FILE = os.path.join(file_path, file_name)
        if not exists(self._LAST_POS_FILE):
            # the first run, we write args.start into the file
            # note that if you started with a non-zero index at the 1st run
            # then I'll only go ahead and resume testing, without checking backward
            # so always pay attention when you assign a magic start value
            self._start: int = input_args.start
            with open(self._LAST_POS_FILE, mode="w+") as f:
                f.write(str(input_args.start))
        else:
            if self._resume:
                # the last index in the file will override args.start
                with open(self._LAST_POS_FILE, mode="r") as f:
                    self._start = int(f.readline().strip())
            else:
                self._start: int = input_args.start

        self._end: int = input_args.end  # 0 indicates to test all apks (default)
        self._logger: Logger = Logger(log_level=input_args.logging)
        self._device_type = "NOT_DEFINED"
        self._overwrite = input_args.overwrite

        path_items = self._apk_folder.split(os.sep)
        self._opt_folder = os.sep.join([*path_items[:-1], f"opt_{path_items[-1]}"])

        self._logger.get_logger.info(f"apk src dir: {self._apk_folder}")
        self._logger.get_logger.info(f"ui otpt dir: {self._opt_folder}")

        if not exists(self._apk_folder):
            self._logger.get_logger.error(f"path not found: {self._apk_folder}")
            exit(1)

        if not exists(self._opt_folder):
            self._logger.get_logger.info(f"mkdir: {self._opt_folder}")
            try:
                mkdir(self._opt_folder)
            except FileNotFoundError:
                self._logger.get_logger.error(f"path not found: {self._opt_folder}")
                exit(1)

        self._apk_list = [a for a in listdir(self._apk_folder) if a.endswith(".apk")]

        # we sort the apk list to keep the consistency when assigning
        # different apk ranges (given by `start` and `end`)
        self._apk_list.sort()
        self.current_index = self._start
        self.dom = None

        try:
            # the adb executable
            adb = "adb"
            # initialize a Device object and connect to the device
            self.device = Device()
            retcode = self.device.connect(self._ip)
            if retcode == 0:
                devices: str = str(check_output([adb, "devices"]))
                if self._ip not in devices:
                    exit(1)

            self._logger.get_logger.debug(f"connected to {self._ip}")
            self._logger.get_logger.info(f"sdk version: {self.device.get_sdk_version()}")
            # init minicap
            self.device.init_minicap()
            self.device.start_minicap()
            # ban auto rotation
            # self.device.run_shell("content insert --uri content://settings/system "
            #                       "--bind name:s:accelerometer_rotation --bind "
            #                       "value:i:0")

            # first, do some clean up jobs
            self.device.stop_3rd_packages()
            self.device.uninstall_3rdpackages()
            self._last_index = 0

        except ConnectionError as ce:
            self._logger.get_logger.error(f"ConnectionError: {ce}")
            exit(1)

    @property
    def apk_total(self) -> int:
        return len(self._apk_list)

    def dump_apks(self, use_package: bool = False) -> int:
        """walk through a path and collectling
        uis at runtime for each app"""
        try:
            t_pre_dump_apks = perf_counter()
            total = len(self._apk_list)
            i = 0
            start_index = int(self._start)
            end_index = len(self._apk_list) if self._end == 0 \
                else max(self._end, len(self._apk_list))

            for i, apk in enumerate(self._apk_list[start_index: end_index]):
                self.current_index = i + start_index
                self._logger.get_logger.info(f"({self.current_index + 1}/{total}) {apk}")
                current_file = join(self._apk_folder, apk)
                try:
                    self.dump_apk(current_file, self.current_index, use_package)
                except:
                    self.device.uninstall_3rdpackages()
                    continue

            t_post_dump_apks = perf_counter()
            self._logger.get_logger.info(
                f'I dump {i} apps in {t_post_dump_apks - t_pre_dump_apks}s')
        except:
            return self.current_index
        return 100000

    def dump_apk(self, apk_file: str, app_index: int, use_package: bool):
        """Install an apk, dump its uis and then uninstall it.
        This function also updates the last_index file
        Args:
            apk_file (str): the apk file path
            app_index (int): the index of current app in all apps
            use_package (bool): use apk's package name to name the output folder
        Returns:
            None (a file will be output to the opt path recording the results)
        """
        apk_pkgname, apk_activity_list = dump_activity_from_apk(apk_file)

        if apk_pkgname == "badfile":
            self._logger.get_logger.warn(f"{apk_file} is a bad zip file")

        else:
            # by default, we take apk hash as the name of output folder
            apk_hash = cal_sha256(apk_file)
            apk_opt_folder = join(self._opt_folder, apk_hash)

            if exists(apk_opt_folder):
                if not self._overwrite:
                    # only skip an app when overwrite mode activated and
                    # the output folder exists
                    self._logger.get_logger.info(
                        f"{apk_file} has been tested, skip")
                    return

            # install the app
            self._logger.get_logger.info(f"now install: {apk_pkgname}")
            t_pre_install = perf_counter()
            code = self.device.install_app(apk_file, apk_pkgname)
            t_post_install = perf_counter()
            if code == 0:
                self._logger.get_logger.debug(f'intall {apk_pkgname} done! '
                                              f'(in {t_post_install - t_pre_install}s)')
                self.device.escape_stuck()  # remove system error float dialogs
                # dump ui
                self.dump_ui(apk_pkgname, apk_activity_list,
                             apk_hash, app_index, use_package)
                # uninstall apk
                code = self.device.uninstall_package(apk_pkgname)
                if code == 0:
                    # update last_index
                    self._last_index += 1
                    with open(self._LAST_POS_FILE, mode='w+') as f:
                        f.write(str(self._last_index))
                else:
                    self._logger.get_logger.warning(
                        f'uninstall {apk_pkgname} fail!')
            else:
                self._logger.get_logger.warning(f'install {apk_pkgname} fail!')

    def drag_hammenu(self):
        print("Dragout Navigation Bar")
        self.device.swipe(0, 0.5, 0.4, 0.5, duration=0.5)

    def dump_ui(self, package_name: str, alist: list,
                apk_hash: str, app_index: int,
                use_package: bool = False):
        """Traverse UIs in an app

        Args:
            package_name (str): Package name declared in the manifest
            alist (list): Activities declared in the manifest
            apk_hash (str): Hash value of the apk file
            use_package (bool): Use package name instead of file hash
              to name the output folder
            app_index (int): Index of the current app in the apk list
        """
        self.device.stop_3rd_packages()
        if use_package:
            # in windows, some prefix are invalid for folder name
            for _invalid_prefix in ['aux', 'com1', 'com2', 'prn', 'con', 'nul']:
                if package_name.startswith(_invalid_prefix):
                    package_name = '_' + package_name
                    break
        out_subfolder = package_name if use_package else apk_hash
        opt_path = join(self._opt_folder, out_subfolder)
        if not exists(opt_path):
            mkdir(opt_path)
        self._logger.get_logger.info(f"Current App: {package_name}")

        alist_focus = set([i for i in alist if is_focus(i)])
        alist_remain = set(alist).difference(alist_focus)
        alist = list(alist_focus)
        alist.extend(list(alist_remain))

        i, k = 0, 0

        # first, extract UIs from its main entrance
        handled_activity = self.extract_ui_from_main(package_name, opt_path, app_index)

        # then, traverse each activity in manifest
        # to access UIs that are missed before
        for a in alist:
            if k == self._FAIL_TRY_TIMES:
                self._logger.get_logger.warning(
                    f'{package_name}: continuous {k} failures, skip')
                break
            i += 1

            self._logger.get_logger.info(
                f"{i}th try for app {app_index}: starting: {a}")

            k = self.handle_ui(package_name, a, k, handled_activity, opt_path)

        self._logger.get_logger.info(f'{package_name}: finished')
        self.device.stop_3rd_packages()

    def extract_ui_from_main(self, package_name: str, opt_path: str,
                             app_index: int) -> set:
        """Launch an app from its main activity, then explore UIs

        Args:
            package_name (str): Package name
            opt_path (str): Output path to store the results
            app_index (int): Index of the current app in the apk list

        Returns:
            A set indicating visited activities
        """
        handled_activity = set()
        self._logger.get_logger.info(f"({app_index}) run app {package_name} from main")
        self.device.start_activity(package_name, "", timeout=1)
        activity = self.device.get_current_activity()["activity"]
        k = self.handle_ui(package_name, activity, 0,
                           handled_activity, opt_path, True)
        if k == 0:  # success
            handled_activity.add(activity)

        # get all interactive controls
        controls = list()

        # we wait up to 5 seconds until an interactive view displays
        # on the main page
        time_delay = 5
        ddl = time.time() + time_delay
        while len(controls) == 0 and time.time() < ddl:
            controls = list()
            self.dom = self.device.get_current_dom()
            if self.dom is not None:
                read_all_nodes(controls, self.dom)
                controls = [self.get_dict(i) for i in controls]
                controls = [i for i in controls if len(i) > 0]
                time.sleep(0.5)

        # main activity
        main_activity = self.device.get_current_activity()["activity"]
        self._logger.get_logger.info(f"main activity: {main_activity}")

        for c in controls:
            try:
                self._logger.get_logger.debug(f"starting: {main_activity}")
                # here, the activity name already contains the package name
                activity = self.device.start_activity(package_name, main_activity)
                self._logger.get_logger.info(f"current activity: {activity}")
                center = (c['c1'], c['c2'])
                self._logger.get_logger.info(f"now click: {c['name']}, "
                                             f"center: {center}, text: {c['text']}")
                self.device.click(*center)
                time.sleep(0.8)
                k = self.handle_ui(package_name, "UNKNOWN",
                                   0, handled_activity, opt_path)
                if k == 0:  # success
                    handled_activity.add(activity)
                    # back to the last activity
                    self.device.press_key(4)
            except:
                continue

        return handled_activity

    @staticmethod
    def get_dict(node) -> dict:
        """Given a hierarchy node, extract the corresponding
        view's name, text, center if it is interactive

        Args:
            node: A node in hierarchy XML

        Returns:
            A dict. When the view is interactive, its 'c1' and 'c2'
              note the center coordinate, and 'name' and 'text' provide
              additional details. Otherwise, return an empty dict
        """
        _dict = dict()
        interact = node.getAttribute('clickable').startswith('t') \
                   or node.getAttribute('long-clickable').startswith('t') \
                   or node.getAttribute('checkable').startswith('t')
        if not interact:
            return _dict

        _dict['name'] = node.getAttribute('class')
        bounds = node.getAttribute('bounds').replace(']', '').split('[')
        if len(bounds) > 1:
            lt = bounds[1].split(',')
            rb = bounds[2].split(',')
            lt = [int(i) for i in lt]
            rb = [int(i) for i in rb]
            _dict['c1'] = int(lt[0] + 0.5 * (rb[0] - lt[0]))
            _dict['c2'] = int(lt[1] + 0.5 * (rb[1] - lt[1]))
        else:
            return dict()

        _dict['text'] = node.getAttribute('text')
        return _dict

    def handle_ui(self, package_name: str, activity: str, k: int,
                  handled_activity: set, opt_path: str,
                  save_control: bool = False) -> int:
        """Given an activity, double-check its displaying on the
        screen, and then take a screenshot and print the hierarchy
        to an XML file

        Args:
            package_name (str): The current package name
            activity (str): The current activity
            k (int): Retry count
            handled_activity (set): Set of visited activities
            opt_path (set): The output path for the apk
            save_control (bool): Parse the hierarchy and extract views

        Returns:
            An int. k + 1 when the process does not finish smoothly
        """
        # start the app
        if activity != "UNKNOWN":
            current_activity = self.device.start_activity(package_name, activity)
        else:
            self._logger.get_logger.info(f"get current activity")
            current_activity = self.device.get_current()[1]
            self._logger.get_logger.info(current_activity)
        if len(current_activity) == 0:
            k += 1
            return k
        if current_activity != activity:
            self._logger.get_logger.info(f'-> {current_activity}')
        if current_activity.count('microvirt'):
            self._logger.get_logger.info('\t[-] activity load: wait fail')
            k += 1
            return k

        if current_activity in handled_activity:
            self._logger.get_logger.info(
                "already seen this activity, skip.")
            return k

        xml_path = os.path.join(opt_path, f"{current_activity}.xml")
        xml_path = xml_path.replace('/', '-')
        self.device.dump_hierarchy(xml_path)
        if not os.path.exists(xml_path):
            k += 1
            return k
        with open(xml_path, mode='r', encoding='utf-8') as xmlf:
            xml = xmlf.read()
        if "package=\"android\"" in xml:
            self.device.press_key(66)  # press enter
            sleep(0.5)
            self.device.press_key(66)
            sleep(0.5)
            self.device.press_key(66)
            self.device.press_key(4)
            self.device.press_key(4)

        has_content, dom = remove_sysnode(xml)
        if not has_content:
            self._logger.get_logger.info(
                '\t[-] without app content, continue')
            os.remove(xml_path)
            k += 1
            return k

        k = 0  # reset k
        self._logger.get_logger.info('\t[*] taking screenshot...')
        try:
            current_activity = current_activity.replace('/', '-')
            image_path = os.path.join(opt_path, f"{current_activity}.jpg")
            self.device.take_screenshot_minicap(image_path)
            with open(xml_path, 'w+', encoding='utf-8') as f:
                dom.writexml(f, addindent='  ', newl='\n')
            if save_control:
                self.dom = dom
            self._logger.get_logger.info('\t[*] save done.')

        except Exception as ex:
            self._logger.get_logger.error(
                f'Exception when saving UI: {ex}')
            k += 1
            return k

        return k


def parse_arg_crawler(input_args: list):
    parser = argparse.ArgumentParser(description="Launch dynamic testing and collect UIs")
    parser.add_argument("apk_folder", type=str,
                        help="the path where you put apks")
    parser.add_argument("ip", type=str,
                        default="127.0.0.1:21523",
                        help="ip address of the android device")
    parser.add_argument("--start", "-s", type=int,
                        help="start index of apk testing",
                        default=0)
    parser.add_argument("--end", "-e", type=int,
                        help="last index of apk testing",
                        default=0)
    parser.add_argument("--resume", "-r", action="store_true",
                        help="if assigned, start from the recorded "
                             "last position instead of value start")
    parser.add_argument("--overwrite", "-o", action="store_true",
                        help="if assigned, overwrite the existing output")
    parser.add_argument("--package_name", "-p", action="store_true",
                        help="if assigned, the output folder for an app will be "
                             "named by the apk's package name instead of its file name")
    parser.add_argument("--logging", "-l",
                        help="logging level, default: info",
                        default="info",
                        choices=["debug", "info", "warn", "error"])
    _args = parser.parse_args(input_args)
    return _args


if __name__ == '__main__':
    from subprocess import getoutput, call

    subprocess.run("taskkill /f /t /im adb")
    args = parse_arg_crawler(sys.argv[1:])
    u = UICrawler(args)
    last_index = u.dump_apks(args.package_name)
    end = u.apk_total if args.end == 0 else max(args.end, u.apk_total)
    while last_index < end:
        args.start = last_index
        # sample code for launch memuc
        # in case of other types of devices, just change the cmd
        index = args.ip[-2]
        o = getoutput(f'memuc isvmrunning -i {index}')
        if not o.lower() == "running":
            print('-------restart-------')
            call(['memuc', 'stop', '-i', index])
            s2 = call(['memuc', 'start', '-i', index])
            if s2 == 0:
                print('done!')
                sleep(10)
        last_index = UICrawler(args).dump_apks(args.package_name)
