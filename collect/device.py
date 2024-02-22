"""Android device object"""

import os.path
import subprocess
import time
from subprocess import CalledProcessError, check_call, check_output
from typing import Union, Tuple
from enum import Enum
import re
from util.util_log import Logger
import airtest

from util.util_xml import remove_sysnode


class Device:
    """An Android device. This class provides APIs to interact with,
    transmit message to, and fetch status from teh device"""

    class Attr(Enum):
        model = "ro.product.model"
        sdk_version = "ro.build.version.sdk"
        cpu_abi = "ro.product.cpu.abi"

    def __init__(self,
                 is_emulator: bool = True,
                 device_name: str = ""):
        self.logger = Logger()

        self.is_emulator = is_emulator
        self.connected = False
        self.device_serial = device_name
        self.display = {}

    @staticmethod
    def get_device_name_by_index(device_index: int = 0) -> Union[str, None]:
        """Get serial by an index in ADB devices"""
        devices = str(check_output("adb devices"))
        i = -2
        for line in devices.split('\n'):
            i += 1
            if i == device_index:
                return line.split('\t')[0].strip()
        return None

    @staticmethod
    def ip_connected(device_ip: str) -> bool:
        """Detect whether an IP is connected with ADB"""
        devices = str(check_output("adb devices"))
        for line in devices.split('\n'):
            if device_ip in line:
                return True
        return False

    def connect(self, device_ip: str) -> int:
        """Connect to an IP via ADB and report the result

        Args:
            device_ip (str): Device IP

        Returns:
            Return code. 0 for success, otherwise the return code
              of the CalledProcessError

        Raises:
            CalledProcessError: Raise when the command fails

        """
        try:
            self.logger.get_logger.info(f"connect to {device_ip} via adb...")
            check_call(["adb", "connect", device_ip])
            devices: str = str(check_output(["adb", "devices"]))
            if device_ip in devices:
                self.connected = True
                self.device_serial = device_ip
            return 0
        except CalledProcessError as e:
            self.logger.get_logger.error(f"cmd {e.cmd} failed: {e.output}")
            return e.returncode

    def init_minicap(self):
        """Initialize Minicap (for screenshot) on the connected device.
        Pull the necessary files according to the Android version if
        they are not on the device, and grant permissions to them.
        """
        root_path = os.path.abspath(os.path.dirname(airtest.__file__))
        abi = self.get_abi()
        android_version = self.get_sdk_version()
        stf_libs = "\\airtest\\core\\android\\static\\stf_libs"
        so_path = f"\\minicap-shared\\aosp\\libs\\android-{android_version}\\{abi}\\minicap.so"
        airtest_minicap_path = os.path.abspath(os.path.dirname(root_path)) + stf_libs
        airtest_minicapso_path = os.path.abspath(os.path.dirname(root_path)) + stf_libs + so_path
        if not os.path.exists(airtest_minicap_path) or \
                not os.path.exists(airtest_minicapso_path):
            raise FileNotFoundError("please check your airtest install")

        dst = "/data/local/tmp/"
        opt = self.run_shell(f"ls /data/local/tmp/minicap", get_output=True)
        if "no such" in opt.lower():
            # push minicap bin and library
            self.logger.get_logger.info("push minicap")
            self.push(f"{airtest_minicap_path}/{abi}/minicap", dst)

        opt = self.run_shell(f"ls /data/local/tmp/minicap.so", get_output=True)
        if "no such" in opt.lower():
            self.push(airtest_minicapso_path, dst)

        # chmod to make minicap executable
        self.run_shell(f"chmod 777 {dst}*")
        self.logger.get_logger.info("minicap files are ready")

    def start_minicap(self, minicap="/data/local/tmp/"):
        """Start Minicap"""
        display_info = self.get_display_info()
        width, height = display_info["width"], display_info["height"]
        self.run_shell(f"LD_LIBRARY_PATH={minicap} "
                       f"{minicap}minicap -P "
                       f"{width}x{height}@{width}x{height}/0 -t")
        self.logger.get_logger.info("minicap is ready")

    def wait_device_ready(self) -> int:
        """Wait for the device until it is online

        Raises:
            CalledProcessError: Raise when the ADB command fails
              or something is wrong with the device
        """
        self.logger.get_logger.debug("waiting device...")
        try:
            check_call(["adb", "-s", self.device_serial, "wait-for-device"])
            self.logger.get_logger.debug("ready")
            return 0
        except CalledProcessError as e:
            self.logger.get_logger.error(f"cmd {e.cmd} failed: {e.output}")
            return e.returncode

    def run_shell(self, shell_cmd: str,
                  get_output: bool = False) -> Union[int, str]:
        """Push ADB shell commands to the device

        Args:
            shell_cmd (str): A command line
            get_output (bool): If shell output is required, set it True

        Returns:
            An int indicating the return code, or the shell output (str)
        """
        cmd = ["adb", "-s", self.device_serial, "shell"]
        shell_cmd = "\"" + shell_cmd + "\""
        cmd.append(shell_cmd)
        cmd = [c for c in cmd if len(c)]
        cmd = ' '.join(cmd)
        try:
            if not get_output:
                # self.logger.get_logger.debug(f"run {cmd}")
                subprocess.run(cmd, timeout=5, capture_output=False, shell=False)
                return 0
            else:
                self.logger.get_logger.debug(f"run {cmd} (getoutput)")
                # timeout may not work with some adb versions (e.g., the memuc adb)
                process = subprocess.run(cmd, timeout=5, capture_output=True, shell=False)

                result = process.stdout.decode()
                if len(result) == 0:
                    result = process.stderr.decode()
                return result
        except CalledProcessError as e:
            self.logger.get_logger.error(f"cmd {e.cmd} failed: {e}")
            devices: str = str(check_output(["adb", "devices"]))
            if self.device_serial not in devices:
                self.connected = False
                raise ValueError(f"device {self.device_serial} is offline")
            else:
                return e.returncode
        except subprocess.TimeoutExpired:
            self.logger.get_logger.info("restart server...")
            subprocess.run(f"adb -s {self.device_serial} kill-server")
            self.logger.get_logger.info("server killed")
            subprocess.run(f"adb -s {self.device_serial} start-server")
            subprocess.run(f"adb connect {self.device_serial}")
            self.logger.get_logger.info("server restarted")
            self.escape_stuck()
            self.run_shell(shell_cmd)

    def take_screenshot_minicap(self, output_file, minicap="/data/local/tmp/"):
        """Take a screenshot by Minicap. The screenshot will be stored
        in `/sdcard/screencap.png` on the device, then it will be pulled
        to the host

        Args:
            output_file (str): The path to store the screenshot on localhost.
              Minicap's jpg encoder will convert the screenshot from png to jpg
              if required
            minicap (str): Remote path for Minicap
        """
        display_info = self.get_display_info()
        width = display_info["width"]
        height = display_info["height"]
        orientation = int(display_info["orientation"]) * 90
        if orientation == 90:
            width, height = height, width
        self.logger.get_logger.debug(f"{width}x{height}@"
                                     f"{width}x{height}/{orientation}")
        self.run_shell(f"LD_LIBRARY_PATH={minicap} {minicap}minicap "
                       f"-P {width}x{height}@{width}x{height}/{orientation} "
                       f"-s > /sdcard/screencap.png")
        self.pull("/sdcard/screencap.png", output_file)

    def get_current(self) -> Tuple[str, str]:
        """Get current package and activity"""
        c = self.run_shell("dumpsys window windows | grep 'Current'",
                           get_output=True)
        c = c.split(' ')[-1].replace('}', '')
        package, activity = c.split('/')
        return package, activity

    def press_key(self, keycode: Union[str, int]):
        self.run_shell(f"input keyevent {keycode}")

    def swipe(self, x0, y0, x1, y1, duration):
        """Swipe the screen
        Ref: droitbot (https://github.com/honeynet/droidbot)
        """
        version = self.get_sdk_version()
        if int(version) <= 15:
            self.logger.get_logger.warning(f"drag: API <= 15 "
                                           f"not supported (current: {version})")
            return
        if int(version) <= 17:
            self.run_shell(f"input swipe {x0} {y0} {x1} {y1}")
        else:
            self.run_shell(f"input touchscreen swipe "
                           f"{x0} {y0} {x1} {y1} {duration}")
        self.logger.get_logger.debug("swipe event sent")

    def get_device_attribute(self, key: str) -> str:
        """Get certain attributes (like sdk version) of the device"""
        if key in dir(self):
            return self.__getattribute__(key)
        ro_key = Device.Attr[key].value
        value = self.run_shell(f"getprop {ro_key}", get_output=True)
        value = value.strip()
        self.__setattr__(key, value)
        return value

    def get_sdk_version(self):
        v = self.get_device_attribute(Device.Attr.sdk_version.name)
        if "\n" in v:
            v = v.split("\n")[-1]
        return v

    def get_model(self):
        return self.get_device_attribute(Device.Attr.model.name)

    def get_abi(self):
        abi = self.get_device_attribute(Device.Attr.cpu_abi.name)
        if "\n" in abi:
            abi = abi.split("\n")[-1]
        return abi

    def get_3rdpackage_installed(self) -> list:
        """list all the installed 3rd-party apps"""
        output = self.run_shell("pm list packages -3", get_output=True)
        packages = re.findall(r'package:([^\s]+)', output)
        return list(packages)

    def get_3rdpackage_running(self) -> list:
        """list all the running 3rd-party apps"""
        self.logger.get_logger.debug("get running 3rd package")
        output = self.run_shell("pm list packages -3", get_output=True)
        packages = re.findall(r'package:([^\s]+)', output)
        process_names = re.findall(r'([^\s]+)$',
                                   self.run_shell("ps; ps -A",
                                                  get_output=True),
                                   re.M)
        return list(set(packages).intersection(process_names))

    def stop_package(self, name: str):
        """Stop a package"""
        code = self.run_shell(f"am force-stop {name}")
        if code != 0:
            self.logger.get_logger.warning(f"stop {name} fails")

    def stop_3rd_packages(self):
        """Stop all 3rd-party packages"""
        targets = self.get_3rdpackage_running()
        for t in targets:
            self.stop_package(t)
        self.logger.get_logger.info("all the 3rd packages are stopped")

    def clear_package_data(self, name: str):
        """Clear data of a package"""
        self.run_shell(f"pm clear {name}")

    def uninstall_package(self, package_name: str) -> int:
        """Uninstall a package

        Returns:
            A return code
        """
        code = self.run_shell(f"pm uninstall {package_name}")
        if code == 0:
            self.logger.get_logger.info(f"uninstall {package_name}: success")
        return code

    def uninstall_3rdpackages(self):
        """Uninstall all 3rd-party packages"""
        self.logger.get_logger.info("uninstalling 3rd packages...")
        for a in self.get_3rdpackage_installed():
            self.uninstall_package(a)
        self.logger.get_logger.info("all the 3rd packages are uninstalled")

    def start_activity(self, package_name: str,
                       activity: str, timeout: int = 0.8) -> Union[str, None]:
        """Try to launch an app with certain activity

        Args:
            package_name (str): The package expected to launch. If it is empty,
              then the activity name will be used instead
            activity (str): The activity we try to access. If it is empty,
              then try to start the package from its default (main) activity.
              We use monkey to access the main activity
            timeout (float): A delay to wait for the activity

        Returns:
            None if fail, an activity name if success
        """
        name = activity
        if len(package_name) > 0:
            name = f"{package_name}/{activity}"
        if name.count("/") > 1:
            name = "/".join(name.split("/")[-2:])
        if len(activity) > 0:
            cmd = f"am start -n {name}"
            self.run_shell(cmd)
        else:
            self.run_shell(f"monkey -p {package_name} -c "
                           "android.intent.category.LAUNCHER 1")
            return None
        ddl = time.time() + timeout
        while time.time() < ddl:
            current_status = self.get_current_activity()
            if current_status is None:
                continue
            if current_status["package"] == package_name:
                break
            time.sleep(.2)
        time.sleep(.2)
        current_status = self.get_current_activity()
        if current_status is None:
            self.logger.get_logger.warning(f"cannot get focus window")
            return ''
        if len(current_status) == 0:
            self.logger.get_logger.warning(f"cannot get focus window")
            return ''
        if current_status["package"] != package_name:
            self.logger.get_logger.warning(f"cannot launch app from "
                                           f"activity {activity}")
            return ''

        # exclude mundane activities (e.g., home)
        if current_status["activity"].count("microvirt"):
            self.logger.get_logger.warning(f"back to android home")
            return ''
        return current_status["activity"]

    def install_app(self, apk_path: str, package_name: str) -> int:
        """Install an app on the device

        Args:
            apk_path (str): An apk file
            package_name (str): The package name of the given apk. It
              is used for confirming the installation

        Returns:
            A return code
        """
        process = subprocess.run(["adb", "-s", self.device_serial, "install",
                                  "-r", "-d", "-g", apk_path], capture_output=True)
        code = process.returncode
        if code == 0:
            apps = self.get_3rdpackage_installed()
            if package_name in apps:
                self.logger.get_logger.info(f"{package_name} installed")
        return code

    def get_display_info(self):
        """Get device display information. This function will
        also get and set width/height of the device object. Value for
        the key named rotation is set to 0 (0), 1 (90), 2 (180), or 3 (270).
        All the results will be saved in `device.display` as well.

        Returns:
            A dict with three keys: orientation, width, and height
        """
        display_re = re.compile(
            r'.*DisplayViewport{valid=true, '
            r'.*orientation=(?P<orientation>\d+), '
            r'.*deviceWidth=(?P<width>\d+), '
            r'deviceHeight=(?P<height>\d+).*'
        )
        output = self.run_shell("dumpsys display", get_output=True).splitlines()
        for line in output:
            m = display_re.search(line, 0)
            if not m:
                continue
            self.display['orientation'] = m.group('orientation')
            self.display['width'] = m.group('width')
            self.display['height'] = m.group('height')
        return self.display

    def pull(self, src: str, dst: str):
        """Pull a file from the device to localhost"""
        try:
            check_call(["adb", "-s", self.device_serial, "pull", src, dst])
        except CalledProcessError:
            return

    def push(self, src: str, dst: str):
        """Push a file to the device from localhost"""
        try:
            check_call(["adb", "-s", self.device_serial, "push", src, dst])
        except CalledProcessError:
            return

    def take_screenshot_screencap(self, save_path: str) -> bool:
        """Take a screenshot via ADB. Note that it is much slower
        than stream-based Minicap"""
        remote_image_path = "/sdcard/screen.png"
        self.run_shell(f"screencap -p {remote_image_path}")
        self.pull(remote_image_path, save_path)
        return True

    def click(self, x, y):
        """Click (tap) on the device screen

        Args:
            x: Horizontal coordinate
            y: Vertical coordinate
        """
        self.run_shell(f"input tap {x} {y}")

    def dump_hierarchy(self, dst):
        """Dump and output the current hierarchy via uiautomator

        Args:
            dst (str): Output path of the hierarchy file
        """
        xml_path = self.run_shell("uiautomator dump", get_output=True)
        xml_path = xml_path.split(' ')[-1].strip()
        if xml_path.endswith('.xml'):
            xml_path = self.run_shell("uiautomator dump", get_output=True)
            xml_path = xml_path.split(' ')[-1].strip()
            self.pull(xml_path, dst)

    def get_current_dom(self):
        """Get current hierarchy DOM

        Returns:
            An XML element. Specifically, the root element of the DOM tree
        """
        try:
            xml_path = "tmp11.xml"
            self.dump_hierarchy(xml_path)
            with open(xml_path, mode='r', encoding='utf-8') as xmlf:
                xml = xmlf.read()
            _, dom = remove_sysnode(xml)
        except FileNotFoundError:
            return None
        os.remove(xml_path)
        return dom

    def get_current_activity(self, retry=3) -> Union[dict, None]:
        """Get current package, PID (process ID) and activity"""
        while retry > 0:
            ret = dict()
            try:
                string = self.run_shell("dumpsys activity top", get_output=True)
            except UnicodeDecodeError:
                self.logger.get_logger.error(
                    "I meet decode error...retry")
                self.escape_stuck()
                retry -= 1
                time.sleep(0.5)
                continue
            re_task = re.compile(r"TASK\s(?P<package>\S+)\s")
            task = re_task.search(string)
            if task:
                ret["package"] = task.group("package")
            else:
                self.logger.get_logger.error("cannot get current package, retry...")
                self.escape_stuck()
                retry -= 1
                time.sleep(0.5)
                continue

            re_activity = re.compile(r"ACTIVITY\s(?P<activity>\S+)"
                                     r"\s\S+\spid=(?P<pid>\d+)")
            activity = re_activity.search(string)
            if activity:
                ret["pid"] = activity.group("pid")
                ret["activity"] = activity.group("activity")
            else:
                self.logger.get_logger.error("cannot get current activity, retry...")
                self.escape_stuck()
                retry -= 1
                time.sleep(0.5)
                continue

            return ret

    def escape_stuck(self):
        """In some cases, the Andoid OS may be freezed (e.g., by
        system built-in dialogs. Then, try to flee from the stuck situation"""
        for _ in range(15):  # greater than max failure
            self.press_key(4)
            time.sleep(0.3)
        time.sleep(1)
        self.press_key(4)
