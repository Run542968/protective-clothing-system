from torchvision.models import resnet18
from torch import nn
from torchvision import transforms
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from conf import conf
import os
from utils import Cropper, softmax
import numpy as np

from special_models import OnnxTsmResnet50, YOLO, ResNet18, DummyModel, C1GPU_TsmResnet50


'''
在本文件中，由于历史遗留原因，"heatmap"及"hm"名称代指不同形式的指示信息
'''


# 只包含单个模型
class SingleModel:
    def __init__(self):
        self.cur_idx = 0
        self.cropper = Cropper()
        self.cur_model = self.load(self.cur_idx)

    def load(self, idx):    # 加载第idx个情景的模型和参数
        if conf.MODEL_COMPLEXITY == 0:
            if conf.STEPS[idx].split()[1] == "手卫生":
                model = OnnxTsmResnet50(os.path.join(conf.MODEL_DIR, "手卫生_tsm.onnx"))
            elif conf.STEPS[idx].split()[1] == '检查穿戴完整性':
                model = DummyModel(20)
            elif conf.STEPS[idx].split()[1] in conf.STEPS_YOLO:
                model = YOLO(os.path.join(conf.MODEL_DIR, "cross-hands-yolov4-tiny.cfg"),
                             os.path.join(conf.MODEL_DIR, "cross-hands-yolov4-tiny.weights"),
                             labels=['hand'], size=256, confidence=conf.MODEL_POSITIVE_THRES_FOR_YOLO
                             )
            else:
                model = ResNet18(os.path.join(conf.MODEL_DIR, conf.MODEL_LST[idx] + '.pkl'))
        elif conf.MODEL_COMPLEXITY == 1:
            if conf.STEPS[idx].split()[1] == '检查穿戴完整性':
                # model = DummyModel(20)
                model = C1GPU_TsmResnet50(os.path.join(conf.MODEL_DIR, conf.MODEL_LST[idx] + '.pkl'), 'common')
            else:
                if conf.STEPS[idx].split()[1] == "手卫生":
                    action_type = 'handwash'
                else:
                    action_type = 'common'
                model = C1GPU_TsmResnet50(os.path.join(conf.MODEL_DIR, conf.MODEL_LST[idx] + '.pkl'), action_type)
        else:
            raise NotImplementedError
        return model

    def forward(self, cv2img):
        info = self.cur_model.forward(cv2img, self.cropper, self.cur_idx)
        # print(f"The current action index:{self.cur_idx+1}, predicted score:{float(info['score'])}, predicted class:{info['pred']}")
        return info

    def update_cur(self, stride=1):
        self.cur_idx += stride
        self.cur_model = self.load(self.cur_idx)


# 包含双模型
class DualModel(SingleModel):
    def __init__(self):
        super(DualModel, self).__init__()
        # nex_model为按流程工作的检测模型，原先的self.cur_model负责检测结果反馈
        self.nex_idx = 0
        self.nex_model = self.load(self.nex_idx)    # 加载第next_idx个情景的模型和参数

    def forward(self, cv2img):
        info = self.nex_model.forward(cv2img, self.cropper, self.nex_idx)      # 下一步骤的分类结果
        if self.cur_idx != self.nex_idx:
            info['hm'] = self.cur_model.forward(cv2img, self.cropper, self.cur_idx)['hm']   # 前一步骤的分类结果
        print(f"The current action index:{self.cur_idx + 1}, predicted score:{float(info['score'])}, predicted class:{info['pred']}")
        return info

    def update_next(self, stride=1):
        self.nex_idx += stride
        self.nex_model = self.load(self.nex_idx)
