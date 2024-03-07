import torchvision.transforms.functional as tF
from conf import conf,load_setting
import numpy as np
import win32gui
import win32con
import ctypes
import cv2
import win32com.client
import os


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def set_window(WINDOW_NAME=None):
    if WINDOW_NAME == None:
        # 解决Windows下应用缩放(DPI Awareness)的问题
        # ref: https://stackoverflow.com/questions/44398075/can-dpi-scaling-be-enabled-disabled-programmatically-on-a-per-session-basis
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        cv2.namedWindow(conf.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(conf.WINDOW_NAME, conf.CANVAS_WIDTH, conf.CANVAS_HEIGHT)
        hwnd = win32gui.FindWindow(None, conf.WINDOW_NAME)
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)     # 最大化窗口
    else:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, conf.CANVAS_WIDTH, conf.CANVAS_HEIGHT)
        hwnd = win32gui.FindWindow(None, WINDOW_NAME)
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)     # 最大化窗口
    # 直接SetForegroundWindow会有bug
    # ref: https://stackoverflow.com/questions/14295337/win32gui-setactivewindow-error-the-specified-procedure-could-not-be-found
    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys('%')
    win32gui.SetForegroundWindow(hwnd)      # 置于最前


def _get_dir_size(path):
    total, info = 0, []
    for file in os.listdir(path):
        size = os.path.getsize(os.path.join(path, file))
        total += size
        info.append((file, size))
    return total, info


def _clean_dir(path, limit):
    size, info = _get_dir_size(path)
    info.sort(key=lambda x: x[0])
    idx = 0
    while size > limit:
        os.remove(os.path.join(path, info[idx][0]))
        size -= info[idx][1]
        idx += 1


class Cropper:
    def __init__(self):
        self.paras = {}
        # (top, left, crop_h, crop_w)
        if not conf.STABLE_VERSION:
            if conf.SCENARIO == 1:
                self.paras[0] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 手卫生
                self.paras[1] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 戴防护帽
                self.paras[2] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 戴口罩
                self.paras[3] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 检查口罩气密性
                self.paras[4] = (conf.FRAME_HEIGHT // 2, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 穿内层鞋套
                self.paras[5] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 戴内层手套
                self.paras[6] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 检查手套气密性
                self.paras[8] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 戴护目镜
                self.paras[9] = (0, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 戴外层手套

                # load_setting(1)
            elif conf.SCENARIO == 2:
                self.paras[0] = (0, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 外层手套手卫生
                self.paras[1] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 脱护目镜
                #self.paras[1] = (0, 0, conf.FRAME_HEIGHT, conf.FRAME_WIDTH)  # 脱护目镜
                self.paras[2] = (0, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 外层手套手卫生
                # tsm的内层手卫生、脱内层手套不做crop
                if conf.MODEL_COMPLEXITY == 0:
                    self.paras[4] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)   # 内层手套手卫生
                # load_setting(2)
            else:
                self.paras[0] = (conf.FRAME_HEIGHT // 2, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 脱内鞋套
                # tsm的内层手卫生、脱内层手套不做crop
                if conf.MODEL_COMPLEXITY == 0:
                    self.paras[1] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 内层手套手卫生
                    self.paras[2] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 脱内层手套
                self.paras[3] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 手卫生
                self.paras[4] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 脱帽子
                # self.paras[5] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 脱口罩
                self.paras[5] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 手卫生
                self.paras[6] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 戴医用外科口罩
                # load_setting(3)

            # for line in conf.Crop_Size_LST:
            #     flash_idx = line[0]
            #     self.paras[flash_idx] = (line[1], line[2], line[3], line[4])
        # STABLE_VERSION
        else:
            if conf.SCENARIO == 1:
                self.paras[0] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 戴防护帽
                self.paras[1] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 手卫生
                self.paras[3] = (0, 0, conf.FRAME_HEIGHT // 3, conf.FRAME_WIDTH)  # 戴护目镜
                self.paras[4] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 戴内层手套
                self.paras[6] = (conf.FRAME_HEIGHT // 2, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 穿内层鞋套
                self.paras[7] = (conf.FRAME_HEIGHT // 2, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 穿外层鞋套
                self.paras[8] = (0, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 戴外层手套

            elif conf.SCENARIO == 2:
                self.paras[0] = (0, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 外层手套手卫生
                self.paras[1] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 脫外层手套
                self.paras[2] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 内层手套手卫生
                self.paras[3] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 内层手套手卫生
                self.paras[5] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 脱内层手套
                self.paras[6] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 脱内层手套
                self.paras[7] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 手卫生

            else:
                self.paras[0] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 戴内层手套
                self.paras[1] = (0, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 脱护目镜
                self.paras[2] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 内层手套手卫生
                self.paras[3] = (0, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 脱口罩
                self.paras[4] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 内层手套手卫生
                self.paras[5] = (0, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 脱帽子
                self.paras[6] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 内层手套手卫生
                self.paras[7] = (conf.FRAME_HEIGHT // 2, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 脱内鞋套
                self.paras[8] = (conf.FRAME_HEIGHT // 2, 0, conf.FRAME_HEIGHT // 2, conf.FRAME_WIDTH)  # 消毒湿巾清洁鞋
                self.paras[9] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 内层手套手卫生
                self.paras[10] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 脱内层手套
                self.paras[11] = (conf.FRAME_HEIGHT // 4, 0, conf.FRAME_HEIGHT // 4, conf.FRAME_WIDTH)  # 手卫生

    def crop(self, img, idx):
        if idx in self.paras.keys():
            top, left, h, w = self.paras[idx]
            img = tF.crop(img, top, left, h, w)
        return img

    def get_crop_pos(self, idx):
        return self.paras[idx]

    def putback(self, cv2img, idx):
        # 把crop出来的部分放回原图上的位置
        if idx in self.paras.keys():
            img = np.zeros((conf.FRAME_HEIGHT, conf.FRAME_WIDTH)).astype(np.uint8)
            top, left, h, w = self.paras[idx]
            img[top: top+h, left: left+w] = cv2img
        else:
            img = cv2img
        return img
