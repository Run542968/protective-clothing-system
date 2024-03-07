from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from conf import conf, load_setting
import os
from utils import set_window, Cropper
from draw_utils import ImageText, _rect_with_rounded_corners, action_lst, rectangle_text, draw_light
import time
from video_recorder import VideoWriter
from tracer import ParallelTracer
import pdb
from video_reader import ThreadCapture


class HandWashAuxiliary:
    def __init__(self):
        self.tracer = ParallelTracer(list(range(1, len(conf.HWD_STEPS))), conf.HW_WINDOW_SIZE, conf.HW_ACTIVATE_THRES)
        self.lst_font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", 35, encoding="utf-8")
        self.rec_font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", 40, encoding="utf-8")

        self.zone_meta = {}
        init_left, init_top = 40, 30
        # top, left, bottom, right，清空区域
        self.zone_meta['clear'] = (855, 570, 1460, 1080)
        # top, left, 项间距, 指示灯直径, 指示灯垂直偏移量
        self.zone_meta['lst'] = (855, 570, 48, 20, 15)
        # 检测结果指示区, top, left, h, w
        self.zone_meta['info'] = (545, 40, 150, int(720 * 0.7))
        # heatmap区域
        self.scale_rate = 0.7  # 摄像画面缩放比例（原宽高分辨率720x1280）
        self.hm_scale_rate = 0.2  # 热力图区缩放比例（原宽高分辨率720x1280）
        self.zone_meta['hm'] = (515 + init_top + int(1280 * self.scale_rate) - int(1280 * self.hm_scale_rate), init_left)

        self.background = (0x00, 0x33, 0x66)
        self.green = (63, 122, 102)
        self.red = (150, 74, 97)
        self.purple = (0x66, 0x66, 0x99)

        self.cur_idx = -1  # 指示最后一颗绿色指示灯的指针
        self.predict = 0
        self.update_light_index = 0

        # 当前热力图
        mat = np.zeros((conf.FRAME_HEIGHT, conf.FRAME_WIDTH)).astype(np.uint8)
        # self.heatmap = cv2.applyColorMap(mat, cv2.COLORMAP_JET)
        self.heatmap = mat



    def init_canvas(self, canvas):
        top, left, bottom, right = self.zone_meta['clear']
        cv2.rectangle(canvas, (left, top), (right, bottom), color=tuple(reversed(self.background)),
                      thickness=cv2.FILLED)
        top, left, gap, diamater, offset_v = self.zone_meta['lst']
        return action_lst(canvas, conf.HWD_STEPS[1:], self.lst_font_style, top, left, gap, diamater, offset_v)

    def draw(self, canvas, cv2img):
        # 更新指示灯
        if self.update_light_index:
            light_idx = self.update_light_index - 1
            # if light_idx < self.cur_idx:
            #     # 做之前的动作，只需将当前灯置绿
            #     canvas = self.set_light(canvas, light_idx, (0, 255, 0))
            # else:
            #     for i in range(self.cur_idx + 1, light_idx):
            #         canvas = self.set_light(canvas, i, (255, 0, 0))     # 中间部分的灯置红
            #     canvas = self.set_light(canvas, light_idx, (0, 255, 0))
            #     self.cur_idx = light_idx
            canvas = self.set_light(canvas, light_idx, (0, 255, 0))
            self.update_light_index = 0

        # jiarun add: heatmap应该一直都有
        canvas = self.heatmap_zone(canvas, cv2img)

        return canvas

    def update_pred(self, predict):
        if type(predict) != int:
            return
        self.predict = predict
        code = self.tracer.trace(predict, ret_type=1)
        if code == -1:
            return
        self.update_light_index = code  # 1-12

    def set_light(self, canvas, idx, color):
        # 设置指示灯颜色
        top, left, gap, diameter, offset_v = self.zone_meta['lst']
        top = top + idx * gap + offset_v

        return draw_light(canvas, top, left, diameter, color)


    def heatmap_zone(self, canvas, cv2img):

        top, left = self.zone_meta['hm']
        color_heatmap = cv2.applyColorMap(self.heatmap, cv2.COLORMAP_JET)
        # print(color_heatmap.shape)
        hm = cv2.resize(color_heatmap, None, fx=self.hm_scale_rate, fy=self.hm_scale_rate)
        cv2img = cv2.resize(cv2img, None, fx=self.hm_scale_rate, fy=self.hm_scale_rate)
        # hm = cv2.resize(color_heatmap, dsize=(self.hm_w, self.hm_h))
        # cv2img = cv2.resize(cv2img, dsize=(self.hm_w, self.hm_h))
        add_img = cv2.addWeighted(cv2img, 0.5, hm, 0.5, 0)
        h, w = cv2img.shape[0:2]
        canvas[top: top + h, left: left + w, :] = add_img
        return canvas


class DrawerVertical:
    def __init__(self, stop_event, skip_event, back_event, dev_mode=False):
        self.scale_rate = 0.7  # 摄像画面缩放比例（原宽高分辨率720x1280）
        self.hm_scale_rate = 0.2  # 热力图区缩放比例（原宽高分辨率720x1280）
        self.hm_h, self.hm_w = 50, 50
        self.touchscreen_enable = conf.TOUCHSCREEN_ENABLE

        # 控制指示灯闪烁
        self.flash_flag = False
        self.flash_timer = 3
        self.flash_idx = 0  # 闪烁的灯指针

        self.cur_idx = 0  # 指向【当前动作指示屏】显示动作的指针
        self.is_finish_detect = False  # 是否属于检测完成状态

        # 当前热力图
        mat = np.zeros((conf.FRAME_HEIGHT, conf.FRAME_WIDTH)).astype(np.uint8)
        # self.heatmap = cv2.applyColorMap(mat, cv2.COLORMAP_JET)
        self.heatmap = mat

        # 示例视频区
        self.example_skip = 2  # 示例视频跳帧播放
        self.example = ThreadCapture(stop_event,
                                     src=os.path.join(conf.EXAMPLE_DIR, conf.EXAMPLE_LST[self.cur_idx] + '.mp4'),
                                     is_loop=True,
                                     buffer_size=self.example_skip)

        # 矩形框文字字号，动作列表文字字号
        self.rec_font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", conf.REC_FONT_SIZE, encoding="utf-8")
        self.lst_font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", conf.LST_FONT_SIZE, encoding="utf-8")

        # 画布区域信息
        self.zone_meta = {}
        init_left, init_top = 40, 30
        # 当前、下一动作提示框
        begin_top, begin_left, rec_h, rec_w = 110 + 435, 556, 130, 484
        self.zone_meta['cur'] = (begin_top, begin_left, rec_h, rec_w)  # top, left, h, w
        begin_top = begin_top + rec_h + 20
        self.zone_meta['next'] = (begin_top, begin_left, rec_h, rec_w)

        # top, left, 项间距, 指示灯直径, 指示灯垂直偏移量 680
        self.zone_meta['lst'] = (855, 570, conf.ITEM_SPACING_SIZE, 20, 10)
        self.zone_meta['cam'] = (init_top + 515, init_left)  # top, left
        self.zone_meta['hm'] = (515 + init_top + int(1280 * self.scale_rate) - int(1280 * self.hm_scale_rate), init_left)
        self.zone_meta['hints'] = (1465, init_left, 430, 1000)  # 要点提示区
        self.zone_meta['examples'] = (20, init_left, 405, 720)  # top, left, h, w
        self.zone_meta['sun_icon'] = (145, 770, 140, 140)  # top, left, h, w
        self.zone_meta['hosp_icon'] = (145, 930, 140, 140)
        self.zone_meta['copyright'] = (820, 330)  # left, top
        # 右上角系统名称 分两行
        self.zone_meta['system_name1'] = (770, 150)
        self.zone_meta['system_name2'] = (820, 210)

        # 使用的颜色
        self.background = (0x00, 0x33, 0x66)
        self.green = (63, 122, 102)
        self.red = (150, 74, 97)
        self.purple = (0x66, 0x66, 0x99)

        # 初始化画布
        self.canvas = np.ones((conf.CANVAS_HEIGHT, conf.CANVAS_WIDTH, 3), np.uint8)
        self.canvas_snapshot = None  # 画布快照

        self.cropper = Cropper()

        self.sleep_time = 1 if conf.FROM_CAMERA_FLAG else 1
        if conf.WRITE_FILE_FLAG:
            self.video_writer = VideoWriter(conf.WRITE_DIR, '{}x{}'.format(conf.CANVAS_WIDTH, conf.CANVAS_HEIGHT))
        else:
            self.video_writer = None
        self.stop_event = stop_event
        self.skip_event = skip_event
        self.back_event = back_event

        # 手卫生细节模块信息显示辅助控制器
        self.handwash = None

        self.init_draw()
        set_window()
        self.dev_mode = dev_mode

    def __del__(self):
        self.example.read(block=False)
        if conf.WRITE_FILE_FLAG:
            self.video_writer.release()
        # if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE):
        #     cv2.destroyWindow(conf.WINDOW_NAME)
        print(f"对象drawer被删除")

    def init_draw(self):
        # 背景色
        self.canvas[:, :, 0] *= self.background[2]
        self.canvas[:, :, 1] *= self.background[1]
        self.canvas[:, :, 2] *= self.background[0]

        # 布局线
        # cv2.rectangle(self.canvas, (34, 34), (conf.CANVAS_WIDTH-34, conf.CANVAS_HEIGHT-34), (0, 0, 0))

        top, left, h, w = self.zone_meta['cur']
        self.rectangle_text(top, left, h, w, '当前动作:\n{}'.format(conf.STEPS[0]), color=self.red)

        top, left, h, w = self.zone_meta['next']
        self.rectangle_text(top, left, h, w, '下一动作:\n{}'.format(conf.STEPS[1]), color=self.purple)

        top, left, gap, diamater, offset_v = self.zone_meta['lst']
        self.action_lst(top, left, gap, diamater, offset_v)

        # 标题文字
        canvas = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)
        font_style = ImageFont.truetype('./fonts/STCAIYUN.TTF', 70, encoding="utf-8")
        draw.text((150, 455), '摄像画面', (255, 255, 255), font=font_style)
        draw.text((660, 455), '流程指导', (255, 255, 255), font=font_style)

        # 版权说明文字
        # left, top = self.zone_meta['copyright']
        # cr_font = ImageFont.truetype("./fonts/STZHONGS.TTF", 20, encoding="utf-8")
        # draw.multiline_text((left, top), '版权所有 ©\n中山大学计算机学院\nAll Rights Reserved.', (255, 255, 255),
        #                     font=cr_font)

        # 系统名称
        left, top = self.zone_meta['system_name1']
        cr_font = ImageFont.truetype("./fonts/STZHONGS.TTF", 33, encoding="utf-8")
        draw.multiline_text((left, top), '新冠肺炎防护服穿脱', (255, 255, 255), font=cr_font)
        left, top = self.zone_meta['system_name2']
        draw.multiline_text((left, top), '智能识别系统', (255, 255, 255), font=cr_font)

        self.canvas = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        # 要点提示
        self.hints_zone(self.cur_idx)

        if conf.STEPS[self.cur_idx].split()[1] == "手卫生":
            # 存储快照，切换手卫生显示控制，快照为手卫生未完成检测的情况
            self.canvas_snapshot = self.canvas.copy()
            # cv2.imshow('1', self.canvas)
            # cv2.waitKey(0)
            self.handwash = HandWashAuxiliary()
            self.canvas = self.handwash.init_canvas(self.canvas)

    def render(self):
        # 将画布渲染输出
        if self.dev_mode:
            cv2.imshow(conf.WINDOW_NAME, self.canvas)
            cv2.waitKey(0)
            return

        if conf.WRITE_FILE_FLAG: # 只录视频，不展示
            self.video_writer.write(self.canvas)
        else:
            if self.touchscreen_enable:
                # jairun: 鼠标回调函数
                user_interaction = {"mouse_clicked": False}
                def mouse_callback(event, x, y, flags, user_interaction):
                    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击
                        (begin_top, begin_left, rec_h, rec_w) = self.zone_meta['next']

                        if begin_left <= x <= begin_left + rec_w and begin_top <= y <= begin_top + rec_h:
                            user_interaction['mouse_clicked'] = True
                            print(f"click skip")
                        else:
                            pass

                cv2.imshow(conf.WINDOW_NAME, self.canvas)
                cv2.setMouseCallback(conf.WINDOW_NAME, mouse_callback, user_interaction)

                code = (cv2.waitKeyEx(self.sleep_time) >> 16)
                if code == 0x25:  # jiarun: 当有按键，并且按键是符合要求的才返回
                    self.back_event.set()
                elif code == 0x27 or user_interaction["mouse_clicked"]:
                    self.skip_event.set()
                if (not conf.WRITE_FILE_FLAG) and cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    # 关闭窗口退出
                    self.stop_event.set()

            else:
                # cv2.imshow(conf.WINDOW_NAME, cv2.rotate(self.canvas, cv2.ROTATE_90_CLOCKWISE)) # jiarun: disabled
                cv2.imshow(conf.WINDOW_NAME, self.canvas)
                code = (cv2.waitKeyEx(self.sleep_time) >> 16)
                # use virtual key code of Windows
                # ref: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
                if code == 0x25:
                    # left arrow
                    self.back_event.set()
                elif code == 0x27:
                    # right arrow
                    self.skip_event.set()
                if (not conf.WRITE_FILE_FLAG) and cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    # 关闭窗口退出
                    self.stop_event.set()

    def draw(self, cv2img):
        # self.handwash = HandWashAuxiliary()
        # self.canvas = self.handwash.init_canvas(self.canvas)
        cv2img2 = cv2img.copy()
        self.camera_zone(cv2img2) # jiarun: 绘制相机画面区域
        self.example_zone() #
        # self.logo_zone()

        # self.canvas = self.handwash.draw(self.canvas)
        if conf.STEPS[self.cur_idx].split()[1] == "手卫生":
            self.canvas = self.handwash.draw(self.canvas, cv2img)
        else:
            self.heatmap_zone(cv2img)
            self.update_flash()
        # self.canvas = cv2.rotate(self.canvas, cv2.ROTATE_90_CLOCKWISE)
        self.render()

    def rectangle_text(self, top, left, h, w, msg, color):
        # 绘制矩形框+文字
        self.canvas = rectangle_text(self.canvas, top, left, h, w, 20, 25, msg, color, self.rec_font_style)

    def action_lst(self, top, left, gap, light_diameter, light_offset_v):
        # 绘制动作列表
        self.canvas = action_lst(self.canvas, conf.STEPS[:-1], self.lst_font_style, top, left, gap,
                                 light_diameter, light_offset_v)

    def set_light(self, idx, color):
        # 设置指示灯颜色
        top, left, gap, diameter, offset_v = self.zone_meta['lst']
        top = top + idx * gap + offset_v

        self.canvas = draw_light(self.canvas, top, left, diameter, color)

    def heatmap_zone(self, cv2img):
        # if len(self.heatmap.shape) == 3 and conf.STEPS[self.cur_idx].split()[1] in conf.STEPS_YOLO:
        #     # 上半部分检测图
        #     # 下半部分原图
        #     # cv2img[0: cv2img.shape[0] // 2, :] = self.heatmap
        #     # cv2img = cv2.resize(cv2img, None, fx=self.scale_rate, fy=self.scale_rate)
        #     # h, w = cv2img.shape[0:2]
        #     # self.canvas[top: top + h, left: left + w, :] = cv2img
        #
        #     hm = cv2.resize(self.heatmap, None, fx=self.scale_rate, fy=self.scale_rate)
        #     h, w = hm.shape[0:2]
        #     self.canvas[top: top + h, left: left + w, :] = hm
        top, left = self.zone_meta['hm']

        if conf.STEPS[self.cur_idx].split()[1] in conf.STEPS_YOLO:
            color_heatmap = np.expand_dims(self.heatmap, axis=2).repeat(3, axis=2)
        else:
            color_heatmap = cv2.applyColorMap(self.heatmap, cv2.COLORMAP_JET)
        # print(color_heatmap.shape)
        hm = cv2.resize(color_heatmap, None, fx=self.hm_scale_rate, fy=self.hm_scale_rate)
        cv2img = cv2.resize(cv2img, None, fx=self.hm_scale_rate, fy=self.hm_scale_rate)
        # hm = cv2.resize(color_heatmap, dsize=(self.hm_w, self.hm_h))
        # cv2img = cv2.resize(cv2img, dsize=(self.hm_w, self.hm_h))
        add_img = cv2.addWeighted(cv2img, 0.5, hm, 0.5, 0)
        h, w = cv2img.shape[0:2]
        self.canvas[top: top + h, left: left + w, :] = add_img

    def camera_zone(self, cv2img):
        top, left = self.zone_meta['cam']
        if self.flash_idx in self.cropper.paras.keys(): # jiarun: 如果当前在闪烁的动作在cropper的列表里，那就按照cropper的参数画一个矩形表示关键区域
            y, x, h, w = self.cropper.paras[self.flash_idx]
            cv2.rectangle(cv2img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=15)
        cv2img = cv2.resize(cv2img, None, fx=self.scale_rate, fy=self.scale_rate) # jiarun: 把摄像画面缩放一下作为显示区域的大小

        h, w = cv2img.shape[0:2]
        self.canvas[top: top + h, left: left + w, :] = cv2img

    def example_zone(self):
        # 示例视频区
        top, left, h, w = self.zone_meta['examples']
        ex = None
        for _ in range(self.example_skip):
            ex = self.example.read()
        try:
            cv2img = cv2.resize(ex, (w, h))
        except:
            cv2img = np.zeros((h, w, 3), np.uint8)
        self.canvas[top: top + h, left:left + w, :] = cv2img

    def get_logo(self, src_img, top, left, h, w):
        assert h == w
        mask = np.zeros((h, w), np.uint8)
        mask_inv = np.ones((h, w), np.uint8)
        r = h // 2
        for i in range(h):
            for j in range(w):
                if (i - r) ** 2 + (j - r) ** 2 < r ** 2:
                    mask[i][j] = 1
                    mask_inv[i][j] = 0

        ground = self.canvas[top: top + h, left: left + w, :]
        img = cv2.resize(src_img, (w, h))
        img_logo = cv2.bitwise_and(img, img, mask=mask)
        img_back = cv2.bitwise_and(ground, ground, mask=mask_inv)
        return cv2.add(img_back, img_logo)

    def logo_zone(self):
        # logo区
        # top, left, h, w = self.zone_meta['compu_icon']
        # ground = self.canvas[top: top + h, left: left + w, :]
        #
        # img = cv2.imread('./icon/logo_compu.png')
        # img = cv2.resize(img, (w, h))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, mask = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        #
        # mask_inv = cv2.bitwise_not(mask)
        #
        # img_logo = cv2.bitwise_and(img, img, mask=mask_inv)
        # img_back = cv2.bitwise_and(ground, ground, mask=mask)
        # self.canvas[top: top + h, left: left + w, :] = cv2.add(img_back, img_logo)

        top, left, h, w = self.zone_meta['sun_icon']
        img1 = cv2.imread('./icon/logo_sun.jpg')
        self.canvas[top: top + h, left: left + w, :] = self.get_logo(img1, top, left, h, w)

        top, left, h, w = self.zone_meta['hosp_icon']
        img2 = cv2.imread('./icon/logo_hosp.jpg')
        self.canvas[top: top + h, left: left + w, :] = self.get_logo(img2, top, left, h, w)

    def hints_zone(self, idx):
        top, left, h, w = self.zone_meta['hints']
        margin = 30

        # 读取文本内容
        try:
            with open(os.path.join(conf.HINTS_DIR, conf.HINTS_LST[idx] + '.txt'), 'r', encoding='utf8') as f:
                text = f.read()
        except:
            text = ""

        canvas = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))

        # 清空区域
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((left, top, left + w, top + h), fill=self.background)

        # 绘制边框及文字
        _rect_with_rounded_corners(canvas, top, left, h, w, 50, 10, (60, 179, 113))
        draw = ImageText(canvas)
        font = './fonts/STZHONGS.TTF'
        # 竖屏 & 新版 -> font_size=33
        draw.write_text_box((left + margin, top + margin), text, box_width=w - margin, font_filename=font,
                            font_size=conf.HINTS_FONT_SIZE, color=(255, 255, 255), line_spacing=conf.HINTS_LINE_SPACING)
        canvas = draw.get_image()
        self.canvas = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def update_flash(self):
        # 更新指示灯闪烁状态
        self.flash_timer -= 1 # jiarun: 过三帧变一下颜色，也就是看起来有闪烁的功能
        if not self.flash_timer: # jiarun: 如果flash_timer=0
            self.flash_timer = 3
            self.flash_flag = not self.flash_flag # flash_flat反转，看起来有闪烁的效果
            color = (255, 0, 0) if self.flash_flag else (128, 128, 128)
            self.set_light(self.flash_idx, color)

    def update_cur(self, stride=1, update_light=False):
        if conf.STEPS[self.cur_idx].split()[1] == "手卫生":
            # 恢复快照，从手卫生切出
            self.canvas = self.canvas_snapshot
            if self.is_finish_detect:
                self.finish_detect(force_light=True)  # 状态同步，对快照切换为检测完成状态
            if stride == 2:
                update_light = True
            # cv2.imshow('1', self.canvas)
            # cv2.waitKey(0)

        self.cur_idx += stride
        top, left, h, w = self.zone_meta['cur']
        self.rectangle_text(top, left, h, w, '当前动作:\n{}'.format(conf.STEPS[self.cur_idx]), color=self.red)
        top, left, h, w = self.zone_meta['next']
        self.rectangle_text(top, left, h, w, '下一动作:\n{}'.format(conf.STEPS[self.cur_idx + 1]), color=self.purple)

        self.example.set_video(src=os.path.join(conf.EXAMPLE_DIR, conf.EXAMPLE_LST[self.cur_idx] + '.mp4'),
                               is_loop=True)
        self.hints_zone(self.cur_idx)
        self.is_finish_detect = False

        if update_light:
            # 是否更新指示灯
            self.update_light() if stride > 0 else self.update_light(False)

        if conf.STEPS[self.cur_idx].split()[1] == "手卫生":
            # 存储快照，切换手卫生显示控制，快照为手卫生未完成检测的情况
            self.canvas_snapshot = self.canvas.copy()
            # cv2.imshow('1', self.canvas)
            # cv2.waitKey(0)
            self.handwash = HandWashAuxiliary()
            self.canvas = self.handwash.init_canvas(self.canvas)

    def update_heatmap(self, info):
        if conf.STEPS[self.cur_idx].split()[1] == "手卫生":
            self.handwash.update_pred(info['pred'])
            self.handwash.heatmap = info['hm']
        elif type(info['hm']) == np.ndarray:
            self.heatmap = info['hm']

    def update_light(self, forward=True):
        # 更新指示灯
        if forward:
            self.set_light(self.flash_idx, (0, 255, 0))
            self.flash_idx += 1
            if self.flash_idx == len(conf.STEPS) - 1:
                self.flash_timer = float('inf')
        else:
            if self.flash_idx < len(conf.STEPS) - 1:
                self.set_light(self.flash_idx, (128, 128, 128))
            self.set_light(self.flash_idx - 1, (128, 128, 128))
            self.flash_idx -= 1
            self.flash_timer = 3

    def finish_detect(self, force_light=False):
        # 当前动作方框变绿，指示灯变绿
        self.is_finish_detect = True
        top, left, h, w = self.zone_meta['cur']
        self.rectangle_text(top, left, h, w, '当前动作:\n{}'.format(conf.STEPS[self.cur_idx]), color=self.green)
        if conf.STEPS[self.cur_idx].split()[1] == "手卫生" and (not force_light):
            return
        self.update_light()

    def reset_detect(self):
        # 当前动作方框变红，指示灯变闪烁 (finish_detect的回滚动作)
        self.is_finish_detect = True
        top, left, h, w = self.zone_meta['cur']
        self.rectangle_text(top, left, h, w, '当前动作:\n{}'.format(conf.STEPS[self.cur_idx]), color=self.red)
        if conf.STEPS[self.cur_idx].split()[1] == "手卫生":
            self.handwash = HandWashAuxiliary()
            self.canvas = self.handwash.init_canvas(self.canvas)
            return
        self.update_light(False)


if __name__ == '__main__':
    load_setting()
    load_setting(1)
    drawer = DrawerVertical(None, None, None, dev_mode=True)
    img = np.zeros((1280, 720, 3), np.uint8)
    drawer.draw(img)

    # img = Image.new("RGB", (720, 1280), (128, 128, 128))
    # # d = ImageDraw.Draw(img)
    # # rounded_rectangle(d, ((100, 100), (700, 700)), 50, outline=(0, 0, 0))
    # _rect_with_rounded_corners(img, 100, 50, (0, 0, 0))
    # img.show()
