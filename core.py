import os
import cv2
import numpy as np
from models import SingleModel, DualModel
from tracer import Tracer
from drawer import Drawer
from drawer_vertical import DrawerVertical
from conf import conf
import time
# import win32com.client
# from win32com.client import constants
import winsound
from video_recorder import VideoWriter
from speaker import Speaker


# 系统结束后的语音播报
end_speeches = ['防护用品穿戴完毕',
                '第一缓冲区内脱防护用品流程结束，脱防护用品流程进入第二缓冲区',
                '您已完成脱防护服的全部步骤，请离开缓冲区，进入清洁区，祝您生活愉快。']


def worker_manual(input_q, output_q, stop_event, skip_event, back_event):
    # 手动切换模型
    model = SingleModel()
    speaker = Speaker()
    cur_idx = 0  # 当前动作序号

    output_q.put({'update': False, 'finish': False, 'reset': False, 'hm': None})  # 哨兵
    speaker.speak('第一个动作 {}'.format(conf.STEPS[0].split()[1]), 'NORMAL', purge=False)
    # 直接播放txt中的动作要点
    text = open(os.path.join(conf.HINTS_DIR, conf.HINTS_LST[0] + '.txt'), encoding='utf-8').read()
    speaker.speak(text, 'NORMAL', purge=False)

    while True:
        d_update_flag = 0
        if stop_event.isSet():
            output_q.put({'update': False, 'finish': False, 'reset': False, 'hm': None})
            break
        frame = input_q.get()
        out = model.forward(frame)
        score = out['score']
        predict = out['pred']
        hm = out['hm']

# ===========================处理跳过命令================================================================================
        if skip_event.isSet():
            if cur_idx == len(conf.STEPS) - 2:
                # 当前检测是最后一个动作时直接结束
                stop_event.set()
                continue
            d_update_flag = 1
            model.update_cur()
            cur_idx += 1
            skip_event.clear()

            speaker.speak('当前动作 {}'.format(conf.STEPS[cur_idx].split()[1]), 'PRIOR')
            text = open(os.path.join(conf.HINTS_DIR, conf.HINTS_LST[cur_idx] + '.txt'), encoding='utf-8').read()
            speaker.speak(text, 'NORMAL', purge=True)

# ===========================处理返回命令================================================================================
        elif back_event.isSet():
            if cur_idx == 0:
                # 当前动作是第一个动作时直接结束
                stop_event.set()
                continue
            model.update_cur(-1)
            d_update_flag = -1
            cur_idx -= 1
            back_event.clear()
            speaker.speak('当前动作 {}'.format(conf.STEPS[cur_idx].split()[1]), 'PRIOR')
            text = open(os.path.join(conf.HINTS_DIR, conf.HINTS_LST[cur_idx] + '.txt'), encoding='utf-8').read()
            speaker.speak(text, 'NORMAL', purge=True)

        output_q.put({'update': d_update_flag, 'finish': False, 'reset': False, 'hm': hm})
    print('Worker thread end ...')


def worker_auto(input_q, output_q, stop_event, skip_event, back_event):
    # 自动切换模型
    model = DualModel()
    tracer = Tracer()
    speaker = Speaker()

    # 给每个动作计时，这里维护一个dict
    FirstFrameFlag = dict(zip(range(0,len(conf.STEPS)-1),[False for _ in range(len(conf.STEPS)-1)])) # jiarun: 字典的内容是{0：False,1：False}，表示是否获得了第一帧
    StartTime = dict()
    EndTime = dict()

    cur_idx = 0  # 当前动作序号
    end_flag = False  # 是否检测到最后一个动作
    end_timer = conf.END_TIMER  # 自动结束定时器 jiarun：程序结束的时候，等待END_TIMER秒然后关闭程序
    # print(f"After procedure ending, wait for {end_timer} seconds.")
    switch_timer = conf.SWITCH_TIMER        # 切换下一动作计时器
    switch_flag = False        # 是否准备进行动作切换

    output_q.put({'update': False, 'finish': False, 'reset': False, 'hm': None})  # 哨兵
    speaker.speak('第一个动作 {}'.format(conf.STEPS[0].split()[1]), 'NORMAL', purge=False)
    # 直接播放txt中的动作要点
    text = open(os.path.join(conf.HINTS_DIR, conf.HINTS_LST[0] + '.txt'), encoding='utf-8').read()
    speaker.speak(text, 'NORMAL', purge=False)

    while True: # jiarun: 保持动作识别线程持续工作
        d_update_flag = 0
        d_finish_flag = False
        d_reset_flag = False
        if stop_event.isSet(): # jiarun: 如果stop被设置，就终止这个动作识别线程
            output_q.put({'update': False, 'finish': False, 'reset': False, 'hm': None})
            break
        frame = input_q.get() # jiarun: 通过Queue完成线程的阻塞，因为初始化的时候队列大小是无限的，所以只有出队的时候会阻塞，当get不到frame的时候就会阻塞
        out = model.forward(frame)
        score = out['score']
        predict = out['pred']
        hm = out['hm']

        # 记录第一次预测当前动作为true的时刻，并把计时保护状态开启
        starttime = time.time()

        if FirstFrameFlag[cur_idx] != True: # 录入信息+开启保护，否则不变: 调试专用 jiarun: 记录当前动作输入第一帧的时间
            StartTime[cur_idx] = starttime
            FirstFrameFlag[cur_idx] = True


# ===========================处理跳过命令================================================================================
        if skip_event.isSet():
            if cur_idx == len(conf.STEPS) - 2:
                # 当前检测是最后一个动作时直接结束

                # 结束之前记录当前时刻是动作完成时刻，如果有开始时间，就记录切换时间为当前动作完成时间；否则不记录，说明还没做这个动作就手动跳过了
                if cur_idx in StartTime.keys():
                    EndTime[cur_idx] = time.time()

                stop_event.set()
                continue
            update_stride = 1 if model.cur_idx == model.nex_idx else 2 # jiarun: 如果cur_idx和next_idx一样，说明这一帧cur_model还没转过来，需要跳两次
            d_update_flag = update_stride
            d_finish_flag = True # jiarun: 直接将动作设置为完成

            model.update_cur(update_stride)

            # 如果有开始时间，就记录切换时间为当前动作完成时间；否则不记录，说明还没做这个动作就手动跳过了
            if cur_idx in StartTime.keys():
                EndTime[cur_idx] = time.time()
            else:
                # pass
                EndTime[cur_idx] = time.time() # 调试的时候使用，用来记录时间
            cur_idx += 1
            FirstFrameFlag[cur_idx] = False # 把切换后的下一动作的Flag更新，使得重新录入StartTime

            tracer.load_tracer(cur_idx) # jiarun: 追踪器追踪新的动作
            model.update_next() # 更新next_model,此时next_model和cur_model的状态是一致的


            switch_flag = False
            switch_timer = conf.SWITCH_TIMER

            skip_event.clear()
            speaker.speak('当前动作 {}'.format(conf.STEPS[cur_idx].split()[1]), 'PRIOR')
            text = open(os.path.join(conf.HINTS_DIR, conf.HINTS_LST[cur_idx] + '.txt'), encoding='utf-8').read()
            speaker.speak(text, 'NORMAL', purge=True)

# ===========================处理返回命令================================================================================
        elif back_event.isSet():
            if cur_idx == 0:
                # 当前动作是第一个动作时直接结束
                stop_event.set()
                continue

            if end_flag:
                # 最后一个动作检测完成的情况特别处理
                d_reset_flag = True
                end_flag = False
                end_timer = conf.END_TIMER
            else:
                if model.cur_idx == model.nex_idx: # jiarun: 如果不等于说明cur_model是处于慢一步的状态
                    d_update_flag = -1
                    model.update_cur(-1)
                else:
                    d_reset_flag = True

                # 切换回上一动作：
                # 移除当前动作和上一动作的key-value pair
                # 把当前动作的Flag改为False，允许重新录入；把上一动作的Flag也改为False，允许重新录入结束时间不用管（会更新）
                FirstFrameFlag[cur_idx] = False
                StartTime.pop(cur_idx, '没有该键(%s)' % (cur_idx))
                EndTime.pop(cur_idx, '没有该键(%s)' % (cur_idx))

                cur_idx -= 1 # jiarun: 回退一步
                FirstFrameFlag[cur_idx] = False
                StartTime.pop(cur_idx, '没有该键(%s)' % (cur_idx))
                EndTime.pop(cur_idx, '没有该键(%s)' % (cur_idx))
                tracer.load_tracer(cur_idx)
                model.update_next(-1)

                switch_flag = False
                switch_timer = conf.SWITCH_TIMER

            back_event.clear()
            speaker.speak('当前动作 {}'.format(conf.STEPS[cur_idx].split()[1]), 'PRIOR')
            text = open(os.path.join(conf.HINTS_DIR, conf.HINTS_LST[cur_idx] + '.txt'), encoding='utf-8').read()
            speaker.speak(text, 'NORMAL', purge=True)

# ============================正常情况===================================================================================
        else:
            # 步骤切换
            # if switch_flag and predict:
            if switch_flag: # jiarun: 当决定switch，只有在这里才算真正切换
                switch_timer -= 1
                if not switch_timer: # jiarun: 当倒计时变为0
                    switch_flag = False
                    switch_timer = conf.SWITCH_TIMER
                    model.update_cur() # cur_model是延迟与next_model的。这是为了展示heatmap
                    tracer.load_tracer(cur_idx)
                    d_update_flag = 1
                    # 播放下一动作的动作要点
                    text = open(os.path.join(conf.HINTS_DIR, conf.HINTS_LST[cur_idx] + '.txt'), encoding='utf-8').read()
                    speaker.speak(text, 'NORMAL', purge=True)

            flag = tracer.trace(predict)        # 检测多帧的窗口中是否为正例

            if flag: # 当整个动作完成的时候，记录时间; 否则空着
                endtime = time.time()
                EndTime[cur_idx] = endtime

            if (not switch_flag) and (not end_flag) and flag:
                try:
                    speaker.speak('已检测到{}动作 下一动作 {}'
                                  .format(conf.STEPS[cur_idx].split()[1], conf.STEPS[cur_idx + 1].split()[1]),
                                  'PRIOR')
                except:
                    speaker.speak('已检测到{}动作 {}'
                                  .format(conf.STEPS[cur_idx].split()[1], end_speeches[int(conf.SCENARIO) - 1]),
                                  'PRIOR')
                print('已检测到{}动作！'.format(conf.STEPS[cur_idx]))
                # 语音重新播报当前动作要点
                # for line in open(os.path.join(conf.HINTS_DIR, conf.HINTS_LST[cur_idx] + '.txt'), encoding='utf-8'):
                #     speak(speaker, line, purge=False)

                d_finish_flag = True
                if cur_idx == len(conf.STEPS) - 2:
                    end_flag = True
                else:
                    cur_idx += 1
                    model.update_next() # jiarun: 先更新next_model，next_model检测下一个动作
                    switch_flag = True

        output_q.put({'update': d_update_flag, 'finish': d_finish_flag, 'reset': d_reset_flag, 'hm': hm})
        if end_flag:
            # 倒计时结束
            end_timer -= 1
            if not end_timer:
                stop_event.set()

    # 保存统计结果
    if not os.path.exists(conf.SUMMARY_PATH):
        os.mkdir(conf.SUMMARY_PATH)
    summary_path = os.path.join(conf.SUMMARY_PATH, 'SCENARIO_' + str(conf.SCENARIO))
    np.save(summary_path + '_StartTime.npy', StartTime)
    np.save(summary_path + '_EndTime.npy', EndTime)
    np.save(summary_path + '_FirstFrameFlag.npy', FirstFrameFlag)
    print("StartTime:", StartTime)
    print("EndTime:", EndTime)
    print("FirstFrameFlag:", FirstFrameFlag)

    print('Worker thread end ...')






def update_drawer(drawer, info):
    if info['finish']:
        drawer.finish_detect()
    if info['update']:
        if conf.AUTO_SWITCH and info['update'] > 0:
            drawer.update_cur(info['update'])
        else:
            drawer.update_cur(info['update'], update_light=True)
    if info['reset']:
        drawer.reset_detect()
    if info['hm'] is not None:
        drawer.update_heatmap(info['hm'])


def gui_main(input_q, output_q, stop_event, skip_event, back_event, cap):
    frame_cnt = 0
    drawer = DrawerVertical(stop_event, skip_event, back_event) if conf.VERTICAL_SCREEN_FLAG else Drawer(stop_event, skip_event, back_event) # jiarun: 初始化gui界面

    if conf.STORAGE_FLAG:
        cam_writer = VideoWriter(conf.STORAGE_PATH, '{}x{}'.format(conf.FRAME_WIDTH, conf.FRAME_HEIGHT),
                                 storage_limit=conf.STORAGE_LIMIT)

    # 异步循环播放背景音乐
    # ref: https://docs.python.org/3/library/winsound.html
    winsound.PlaySound(conf.BACKGROUND_MUSIC_PATH, winsound.SND_ASYNC | winsound.SND_FILENAME | winsound.SND_LOOP)

    while True: # jiarun: 保证这个gui线程一直alive
        # start = time.perf_counter()
        if stop_event.isSet():
            input_q.put(np.zeros((conf.FRAME_HEIGHT, conf.FRAME_WIDTH, 3), np.uint8)) # jiarun: 如果stop_event是True, 那就往input_queue送入空白帧，并且中断gui线程的执行
            break

        try:
            ret, frame = cap.read() # jiarun: 没有stop就从摄像头读取视频帧

            if not ret:
                input_q.put(np.zeros((conf.FRAME_HEIGHT, conf.FRAME_WIDTH, 3), np.uint8))
                stop_event.set() # jiarun: 中断线程事件被触发
                print(f"当前配置下，视频帧来自：{'camera' if conf.FROM_CAMERA_FLAG else 'example_video'}。ret:{ret}，视频文件读帧完毕，程序终止")
                break
        except:
            raise IOError(f"当前配置下，视频帧来自：{'camera' if conf.FROM_CAMERA_FLAG else 'example_video'}, 读取帧异常。")

        if conf.FROM_CAMERA_FLAG:
            # 摄像头裁剪+旋转
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # jiarun: 旋转保证输入的摄像头的广角，这样能拍全人体
            frame = cv2.flip(frame, 1) # jiarun: 翻转克服镜像
            scale = conf.CAP_HEIGHT // 16 # jiarun-NOTE: 手动设置CAP_HEIGHT的目的是，如果frame大于这个高度，就按照这个高度中心裁剪，如果小于这个分辨率，就按照摄像头的分辨率。不过还是根据摄像头自动捕捉比较合理，不用手动设置，手动设置会因为摄像头的协议而失效，罗技摄像头的分辨率就改不了
            crop_h, crop_w = scale * 16, scale * 9 # jiarun: 把摄像头输入的画面安装16:9给裁剪一下
            top = conf.CAP_HEIGHT // 2 - crop_h // 2
            left = conf.CAP_WIDTH // 2 - crop_w // 2
            frame = frame[top: top + crop_h, left: left + crop_w]
        else:
            # 文件输入流裁剪
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # jiarun: 本来是W,H，旋转后变成H,W
            frame = cv2.flip(frame, 1)
            scale = conf.CAP_HEIGHT // 16
            crop_h, crop_w = scale * 16, scale * 9 # jiarun: 16:9的比例，是一种中心裁剪
            top = conf.CAP_HEIGHT // 2 - crop_h // 2
            left = conf.CAP_WIDTH // 2 - crop_w // 2
            frame = frame[top: top + crop_h, left: left + crop_w]
        frame = cv2.resize(frame, (conf.FRAME_WIDTH, conf.FRAME_HEIGHT)) # jiarun: 把中心裁剪好的帧resize成1280*720的像素
        if conf.STORAGE_FLAG:
            cam_writer.write(frame)
        if frame_cnt % conf.DETECT_INTERVAL == 0: # jiarun: 不需要每一帧都送进入识别，间隔4帧采一帧，并更新画面
            input_q.put(frame.copy())
            info = output_q.get()  # 取出上一次检测结果 jiarun: 当get不到结果的时候就会阻塞
            update_drawer(drawer, info) # 根据检测信息更新画面
        # start = time.perf_counter()
        if frame_cnt % conf.DISPLAY_INTERVAL == 0: # jiarun: 间隔多少帧绘制一次画面
                drawer.draw(frame)
        # end = time.perf_counter()
        frame_cnt += 1
        # end = time.perf_counter()
        # print(end - start)

    # 停止播放背景音乐
    winsound.PlaySound(None, winsound.SND_PURGE)

    print('GUI thread end ...')
    # cap.release()

    cv2.destroyAllWindows()





