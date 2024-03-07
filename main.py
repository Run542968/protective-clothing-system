import os
import time
import datetime


from comtypes.client import CreateObject, Constants
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from threading import Thread, Event
from threading import enumerate as thread_enumerate
from queue import Queue
from core import worker_manual, worker_auto, gui_main
from conf import conf, load_setting
from utils import set_window
import tkinter.ttk as ttk
import pyperclip
from draw_statistic import Draw_Statistic
import torch





def draw_cover(kb_ret, exp_date, touchscreen_enable):
    # 绘制封面
    canvas = np.ones((conf.CANVAS_HEIGHT, conf.CANVAS_WIDTH, 3), np.uint8) # jiarun: (H,W,3)
    rgb = [0x00, 0x33, 0x66]
    canvas[:, :, 0] *= rgb[2]
    canvas[:, :, 1] *= rgb[1]

    if conf.VERTICAL_SCREEN_FLAG:
        s1, s2, s3, s4 = 80, 50, 20, 40 # jiarun-NOTE: 字号
        p1, p2, p3, p4, p5 = (75, 520), (250, 750), (310, 980), (34, 1350), (150, 1200) # jiarun-NOTE: 组件的位置
    else:
        s1, s2, s3, s4 = 120, 80, 20, 40
        p1, p2, p3, p4, p5 = (200, 200), (530, 450), (550, 700), (34, 930), (530, 900)

    font_style = ImageFont.truetype("./fonts/STXINGKA.TTF", s1, encoding="utf-8")
    canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    draw.text(p1, '新冠肺炎防护过程识别系统', (255, 255, 255), font=font_style)

    font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", s2, encoding="utf-8")
    draw.text(p2, '当前模式: {}'.format('自动模式' if conf.AUTO_SWITCH else '手动模式')
              , (255, 255, 255), font=font_style)
    draw.text(p3, '请按任意键启动 ...', (255, 255, 255), font=font_style)

    # font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", s3, encoding="utf-8")
    # draw.multiline_text(p4, '版权所有 ©\n中山大学计算机学院\nAll Rights Reserved.', (255, 255, 255),
    #                     font=font_style)

    font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", s4, encoding="utf-8")
    draw.text(p5, '许可证有效期至: {}'.format(exp_date), (255, 255, 255), font=font_style)

    canvas = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    set_window()

    if touchscreen_enable:
        # jairun: 鼠标回调函数
        mouse_click = {"click":False}
        def mouse_callback(event, x, y, flags, mouse_click):
            if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击
                mouse_click['click'] = True
        cv2.setMouseCallback(conf.WINDOW_NAME, mouse_callback, mouse_click)

        cv2.imshow(conf.WINDOW_NAME, canvas)

        while True:
            if cv2.waitKey(1) & 0xFF != 0xFF or mouse_click['click']:
                kb_ret.append(0)
                break

            if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print(f"窗口在封面被关闭，窗口状态：{cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE)}，终止程序...")
                break
    else:
        cv2.imshow(conf.WINDOW_NAME, canvas)
        while True:
            if cv2.waitKey(1) & 0xFF != 0xFF: # jiarun: 当有按键，并且按键是符合要求的才返回
                kb_ret.append(0)
                break
            if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print(f"窗口在选择情景的时候被关闭，窗口状态：{cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE)}，终止程序...")
                break

    if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE):
        cv2.destroyWindow(conf.WINDOW_NAME)


def draw_menu(kb_ret, touchscreen_enable):
    # 绘制初始菜单
    canvas = np.ones((conf.CANVAS_HEIGHT, conf.CANVAS_WIDTH, 3), np.uint8)
    rgb = [0x00, 0x33, 0x66]
    canvas[:, :, 0] *= rgb[2]
    canvas[:, :, 1] *= rgb[1]
    canvas[:, :, 2] *= rgb[0]

    if conf.VERTICAL_SCREEN_FLAG:
        s1, s2 = 80, 20
        p1, p2, p3, p4, p5 = (70, 550), (70, 750), (70, 950), (80, 1150), (34, 1350)
        rec2, rec3, rec4 = (70+480,750+125), (70+960,950+125), (70+960,1150+125) # rigth_down_point
    else:
        s1, s2 = 90, 20
        p1, p2, p3, p4, p5 = (300, 100), (300, 350), (300, 600), (300, 850), (34, 930) # left_top_point
        rec2, rec3, rec4 = (300+540,350+125), (300+1070,600+125), (300+1070,850+125) # rigth_down_point

    font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", s1, encoding="utf-8")
    canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    draw.text(p1, '请选择使用情景 ...', (255, 255, 255), font=font_style)
    draw.text(p2, '情景1 穿流程', (255, 255, 255), font=font_style)
    draw.text(p3, '情景2 脱流程 (第一缓冲区)', (255, 255, 255), font=font_style)
    draw.text(p4, '情景3 脱流程 (第二缓冲区)', (255, 255, 255), font=font_style)

    # cr_font = ImageFont.truetype("./fonts/STZHONGS.TTF", 20, encoding="utf-8")
    # draw.multiline_text(p5, '版权所有 ©\n中山大学计算机学院\nAll Rights Reserved.', (255, 255, 255),
    #                     font=cr_font)

    canvas = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR) # jiarun: 转换nparray的RGB格式
    # jiarun: for adjust layout
    # cv2.rectangle(canvas, p2, rec2, color=(0, 0, 255), thickness=4)
    # cv2.rectangle(canvas, p3, rec3, color=(0, 0, 255), thickness=4)
    # cv2.rectangle(canvas, p4, rec4, color=(0, 0, 255), thickness=4)
    set_window() # jiarun: enabel自适应窗口大小，并且能够拖动放缩窗口


    if touchscreen_enable:
        # jairun: 鼠标回调函数
        user_interaction = {"mouse_clicked": None}
        def mouse_callback(event, x, y, flags, user_interaction):
            if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击
                if p2[0] <= x <= rec2[0] and p2[1] <= y <= rec2[1]:
                    user_interaction['mouse_clicked'] = 1
                    print(f"click p2")
                elif p3[0] <= x <= rec3[0] and p3[1] <= y <= rec3[1]:
                    user_interaction['mouse_clicked'] = 2
                    print(f"click p3")
                elif p4[0] <= x <= rec4[0] and p4[1] <= y <= rec4[1]:
                    user_interaction['mouse_clicked'] = 3
                    print(f"click p4")
                else:
                    pass

        cv2.imshow(conf.WINDOW_NAME, canvas)
        cv2.setMouseCallback(conf.WINDOW_NAME, mouse_callback, user_interaction)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 0xFF and key in (ord('1'), ord('2'), ord('3')): # jiarun: 当有按键，并且按键是符合要求的才返回
                kb_ret.append(chr(key))
                break
            if user_interaction["mouse_clicked"] != None:
                kb_ret.append(str(user_interaction["mouse_clicked"]))
                break
            if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print(f"窗口在选择情景的时候被关闭，窗口状态：{cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE)}，终止程序...")
                break
    else:
        cv2.imshow(conf.WINDOW_NAME, canvas)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 0xFF and key in (ord('1'), ord('2'), ord('3')): # jiarun: 当有按键，并且按键是符合要求的才返回
                kb_ret.append(chr(key))
                break
            if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print(f"窗口在选择情景的时候被关闭，窗口状态：{cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE)}，终止程序...")
                break


    if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE):
        cv2.destroyWindow(conf.WINDOW_NAME) # jiarun: 销毁窗口


def draw_statistic(kb_ret, scenario , cellTexts, actions, most_rate, touchscreen_enable):

    # 绘制封面
    canvas = np.ones((conf.CANVAS_HEIGHT, conf.CANVAS_WIDTH, 3), np.uint8)
    rgb = [0x00, 0x33, 0x66]
    canvas[:, :, 0] *= rgb[2]
    canvas[:, :, 1] *= rgb[1]

    if conf.VERTICAL_SCREEN_FLAG:
        s1, s2, s3, s4 = 120, 60, 28, 40
        p1, p2, p3, p4, p5 = (300, 180), (420, 380), (20, 540), (420, 1680), (400, 1780)
    else:
        s1, s2, s3, s4 = 120, 50, 23, 40
        p1, p2, p3, p4, p5 = (700, 80), (100, 220), (150, 340), (460, 950), (910, 1000)

    font_style = ImageFont.truetype("./fonts/STXINGKA.TTF", s1, encoding="utf-8")
    canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    draw.text(p1, '总结报告', (255, 255, 255), font=font_style)

    font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", s2, encoding="utf-8")
    draw.text(p2, '用时统计: ', (255, 255, 255), font=font_style)

    font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", s3, encoding="utf-8")
    draw.multiline_text(p3,
                        '{0:{5}^10} \t {1:{5}^8} \t {2:{5}^5} \t {3:{5}^5} \t {4:{5}^5}'.format('动作名称',' 开始时刻', '结束时刻', '总用时' , '评分', chr(12288)),
                        (255, 255, 255),
                        font=font_style)

    margin = (1150-600)/len(actions) if conf.VERTICAL_SCREEN_FLAG else (950-350)/len(actions)-15
    for i in range(len(actions)):
        w, h = p3[0], p3[1]+(i+1)*margin
        draw.multiline_text((w,h),
                            '{0:{5}^12} \t {1:^13} \t {2:^14} \t {3:^13} \t {4:{5}^4}'.format(actions[i].strip('\n'),cellTexts[i][0],cellTexts[i][1],cellTexts[i][2],cellTexts[i][3],chr(12288)),
                            (255, 255, 255),
                            font=font_style)

    # 直方图
    left, top, w, h = (80, 1210, 420, 420) if conf.VERTICAL_SCREEN_FLAG else (1300, 250, 400, 400)
    img_line = cv2.imread('./summary/SCENARIO_' + str(scenario) + '_line.png')
    img_line = cv2.resize(img_line, (w, h))
    img_line = Image.fromarray(img_line)
    canvas.paste(img_line, (left, top))

    # 饼图
    left, top, w, h = (600, 1210, 420, 420) if conf.VERTICAL_SCREEN_FLAG else (1300, 660, 400, 400)
    img_pie = cv2.imread('./summary/SCENARIO_' + str(scenario) + '_pie.png')
    img_pie = cv2.resize(img_pie,(w,h))
    img_pie = Image.fromarray(img_pie)
    canvas.paste(img_pie,(left,top))

    font_style = ImageFont.truetype("./fonts/STXINGKA.TTF", s2, encoding="utf-8")
    draw.text(p4, '总评：{}'.format(most_rate[0][0]), (255, 255, 255), font=font_style)

    font_style = ImageFont.truetype("./fonts/STZHONGS.TTF", s3, encoding="utf-8")
    draw.text(p5, '请按任意键关闭 ...', (255, 255, 255), font=font_style)


    canvas = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    # cv2.imshow(conf.WINDOW_NAME, canvas)

    set_window(conf.WINDOW_NAME)

    if touchscreen_enable:
        # jiarun: 鼠标回调函数
        mouse_click = {"click":False}
        def mouse_callback(event, x, y, flags, mouse_click):
            if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击
                mouse_click['click'] = True
                print(f"click statistic")

        cv2.imshow(conf.WINDOW_NAME, canvas)

        cv2.setMouseCallback(conf.WINDOW_NAME, mouse_callback, mouse_click)


        while True:
            if cv2.waitKey(1) & 0xFF != 0xFF or mouse_click['click']:
                kb_ret.append(0)
                break

            if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print(f"窗口在统计被关闭，窗口名：{conf.WINDOW_NAME}, 窗口状态：{cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE)}，终止程序...")
                break
    else:
        cv2.imshow(conf.WINDOW_NAME, canvas)
        if cv2.waitKey(0) != -1:
            kb_ret.append(0)

    if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE):
        cv2.destroyWindow(conf.WINDOW_NAME)


def draw_alert(msg, code):
    # 绘制授权警告窗口
    def copy_to_clip_board():
        pyperclip.copy(code)

    window = tk.Tk()
    window.title('权限错误')
    window.geometry('400x170+400+200')

    la = tk.Label(window,
                  text='{}\n\n请联系管理员，并提供以下本机识别码:'.format(msg),  # 标签的文字
                  font=('Song', 12), pady=10)  # 标签长宽
    la.pack()  # 固定窗口位置
    text = tk.Text(window, height=3, width=55, spacing2=5)
    text.bind("<Key>", lambda e: "break")  # 阻止用户输入
    text.pack()
    text.insert(tk.INSERT, code)

    btn1 = ttk.Button(window, text='确定', command=window.destroy)
    btn1.pack(pady=5, side='left', padx=(100, 0))
    btn2 = ttk.Button(window, text='复制', command=copy_to_clip_board)
    btn2.pack(pady=5, side='right', padx=(0, 100))
    window.mainloop()


def openCap():
    s = time.time()
    if conf.FROM_CAMERA_FLAG:
        try:
            cap = cv2.VideoCapture(conf.CAMERA_ID, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, conf.CAP_HEIGHT) # jiarun-NOTE: 这两个set确实没用，离谱，应该是摄像头协议的原因
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, conf.CAP_WIDTH)
            # conf.CAP_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # conf.CAP_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            status, frame = cap.read() # 尝试读取一帧，看看是否正常，不正常就要换摄像头ID
            if not status:
                print(f"摄像头ID不正确，请在setting.json5文件修改摄像头ID,即CAMERA_ID")
                return None
        except:
            print(f"cv2摄像头启动失败")
    else:
        try:
            cap = cv2.VideoCapture(os.path.join(conf.VIDEO_DIR, conf.VIDEO_NAME))
        except:
            print(f"cv2视频读取失败")
    e = time.time()
    print('摄像头启动时间：', e - s)

    return cap


def main(cap):
    exp_date = datetime.datetime.strptime("2100-01-01", "%Y-%m-%d") # jiarun-NOTE: added

    speaker = CreateObject("SAPI.SpVoice")
    constants = Constants(speaker)
    # speaker = win32com.client.gencache.EnsureDispatch("SAPI.SpVoice")
    # speaker = win32com.client.Dispatch("SAPI.SpVoice")
    print('系统已就绪 ...\n')
    stop_event = Event()  # 是否停止检测
    skip_event = Event()  # 是否跳过当前动作
    back_event = Event()  # 是否回退上一动作

    # 1.封面状态
    kb_ret = []
    t = Thread(target=draw_cover, args=(kb_ret, exp_date, conf.TOUCHSCREEN_ENABLE))
    t.start()

    if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE):
        cv2.destroyWindow(conf.WINDOW_NAME)
    t.join()

    if len(kb_ret) == 0 : # jiarun: kb_ret是空的，说明遇到了不正常的返回，例如窗口被手动关闭了，直接结束程序就行
        return False


    # 2.选择情景
    kb_ret = []
    t = Thread(target=draw_menu, args=(kb_ret, conf.TOUCHSCREEN_ENABLE))
    t.start()
    if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE):
        cv2.destroyWindow(conf.WINDOW_NAME)
    t.join()

    if len(kb_ret) == 0: # jiarun: kb_ret是空的，说明遇到了不正常的返回，例如窗口被手动关闭了，直接结束程序就行
        return False
    else:
        scenario = kb_ret[0]

    try:
        scenario = int(scenario[-1])
    except:
        to_number = {'一': 1, '二': 2, '三': 3}
        scenario = to_number[scenario[-1]]
    speaker.Speak('使用情景 {}'.format(scenario), constants.SVSFlagsAsync | constants.SVSFPurgeBeforeSpeak)
    print('使用情景: {}'.format(scenario))

    # 3. 正式的场景界面
    # 导语
    start_speeches = ['您好，您现在位于穿防护服区域，请按照以下标准流程操作。',
                      '您好，您现在位于第一缓冲区，请按照以下标准流程操作。',
                      '您好，您现在位于第二缓冲区，请按照以下标准流程操作。']
    speaker.Speak(start_speeches[scenario - 1], constants.SVSFlagsAsync)

    load_setting(scenario)  # 加载配置
    input_q = Queue()  # camera工作队列
    output_q = Queue() # model工作队列

    # jiarun: 从下面开始就两个子线程了，一个gui_thread，一个是action_recognizer_thread, 他们之间用queue进行阻塞和通信，实现异步操作
    if conf.AUTO_SWITCH: # jiarun: 加载的动作识别模型是不是自动切换的（也就是检测到一个动作自动切到下一个）
        work_t = Thread(target=worker_auto, args=(input_q, output_q, stop_event, skip_event, back_event))
    else:
        work_t = Thread(target=worker_manual, args=(input_q, output_q, stop_event, skip_event, back_event))
    gui_t = Thread(target=gui_main, args=(input_q, output_q, stop_event, skip_event, back_event, cap))
    work_t.start()
    gui_t.start()


    work_t.join()
    gui_t.join()

    speaker.Speak('检测结束,等待总结报告生成', constants.SVSFlagsAsync | constants.SVSFPurgeBeforeSpeak)
    print('检测结束,等待总结报告生成...')

    # 4. 展示总结报告
    if conf.SUMMARY_ENABLE:
        kb_ret = []
        cellTexts, actions, most_rate = Draw_Statistic(scenario)
        t = Thread(target=draw_statistic, args=(kb_ret, scenario, cellTexts, actions, most_rate, conf.TOUCHSCREEN_ENABLE))
        t.start()

        if cv2.getWindowProperty(conf.WINDOW_NAME, cv2.WND_PROP_VISIBLE):
            cv2.destroyWindow(conf.WINDOW_NAME)
        t.join()

        if not len(kb_ret):
            return False


    print('检测已结束 ... 系统复位中 ...')
    return True

def check_gpu(use_gpu):
    cuda_is_available = torch.cuda.is_available()
    if cuda_is_available:
        if not use_gpu:
            print(f"当前设备有独显GPU,可在setting.json5设置USE_GPU=True获得更好的检测效果。")
    else:
        if use_gpu:
            print(f"当前设备无独显GPU,请在setting.json5设置USE_GPU=false。")
            return False
    return True

if __name__ == '__main__':

    load_setting(0)

    # 自适应的适配横屏和竖屏
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    if (screen_width<screen_height):
        conf.VERTICAL_SCREEN_FLAG = True
    else:
        conf.VERTICAL_SCREEN_FLAG = False

    # jiarun: 加载一些默认config，这些参数不需要写在setting.json里
    conf.MODEL_COMPLEXITY = 1
    conf.MODEL_DIR = conf.C1_MODEL_DIR
    conf.STABLE_VERSION = False
    conf.CANVAS_HEIGHT = 1920 if conf.VERTICAL_SCREEN_FLAG else 1080
    conf.CANVAS_WIDTH = 1080 if conf.VERTICAL_SCREEN_FLAG else 1920
    conf.CAP_HEIGHT = 1920
    conf.CAP_WIDTH = 1080
    conf.FRAME_HEIGHT = 1280
    conf.FRAME_WIDTH = 720


    Cap = openCap()     # 摄像头
    if Cap != None: # jiarun: 摄像头用对了才进入下一步
        if check_gpu(conf.USE_GPU):  # jiarun: 检测能不能用GPU
            ret = True
            while ret:
                ret = main(Cap)
                print('当前存活线程: {}'.format(len(thread_enumerate())))

            Cap.release()

