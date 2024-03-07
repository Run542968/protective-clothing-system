import json5
import os


class Dict(dict):
    # 将字典转化为可通过'.'访问属性的方式
    def __getattr__(self, key):
        return self.get(key)


conf = Dict()


def load_setting(scenario=0):
    # 功能性配置
    if conf.STABLE_VERSION and scenario != 0:
        file_path = 'settings_stable.json5'
        index = scenario - 1
    else:
        file_path = 'settings.json5'
        index = scenario

    with open(file_path, 'r', encoding='utf-8') as fp:
        info = json5.load(fp)
    conf.update(info[index])

    # # 调整部分配置
    # if scenario == 0:
    #     if conf.MODEL_COMPLEXITY == 1:
    #         conf.MODEL_DIR = conf.C1_MODEL_DIR  # 更换权重文件夹
    #     return

    # 样式配置
    style_root_path = './style'
    if conf.STABLE_VERSION and conf.VERTICAL_SCREEN_FLAG:
        style_file = 'style_stable_vertical.json5'
    elif conf.STABLE_VERSION and not conf.VERTICAL_SCREEN_FLAG:
        style_file = 'style_stable_horizontal.json5'
    elif not conf.STABLE_VERSION and conf.VERTICAL_SCREEN_FLAG:
        style_file = 'style_new_vertical.json5'
    else:
        style_file = 'style_new_horizontal.json5'

    with open(os.path.join(style_root_path, style_file), 'r', encoding='utf-8') as fp2:
        info2 = json5.load(fp2)
    conf.update(info2[scenario - 1])
