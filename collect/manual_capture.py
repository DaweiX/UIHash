"""A simple script to test the capture function"""

from os.path import exists, join, abspath
from os import makedirs
import sys, subprocess


class ManualCapture:
    def __init__(self, save_path: str, ip: str) -> None:
        self.save_path = save_path
        if not exists(self.save_path):
            makedirs(self.save_path)
        print("save path:", self.save_path)
        self.device = Device()
        self.device.connect(ip)

    def capture(self):
        current_activity = self.device.get_current()[1].strip()
        print(f"now: {current_activity}", end="...")
        xml_path = join(self.save_path, f"{current_activity}.xml")
        current_activity = current_activity.replace('/', '-')
        self.device.dump_hierarchy(xml_path)
        with open(xml_path, mode='r', encoding='utf-8') as xmlf:
            xml = xmlf.read()

        has_content, dom = remove_sysnode(xml)
        if not has_content:
            print("no content in the dom")
            return
        try:
            current_activity = current_activity.replace('/', '-')
            image_path = join(self.save_path, f"{current_activity}.jpg")
            self.device.take_screenshot_minicap(image_path)
            with open(xml_path, 'w+', encoding='utf-8') as f:
                dom.writexml(f, addindent='  ', newl='\n')

            print(f'save done!')

        except Exception as ex:
            print(ex)

if __name__ == "__main__":
    workpath = abspath(join(__file__, "..", ".."))
    print("work path:", workpath)
    sys.path.append(workpath)
    if len(sys.argv) < 3:
        print("please call me with two paras (1: save path, 2: device ip)")
        exit(0)
    path, ip = sys.argv[1], sys.argv[2]
    from device import Device
    from util.util_xml import remove_sysnode
    subprocess.run("taskkill /f /t /im adb")
    mc = ManualCapture(path, ip)
    key = input("Press Enter to capture")
    while key == '':
        mc.capture()
        key = input("Press Enter to capture")
    