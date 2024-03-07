from threading import Thread, Lock, Event
from queue import Queue
import cv2


# 独立线程视频解码器
class ThreadCapture:
    def __init__(self, stop_event:Event, src=None, cap=None, is_loop=False, buffer_size=1):
        self.inner_lock = Lock()
        self.stop_event = stop_event
        self.set_video(src, cap, is_loop)
        self.queue = Queue(maxsize=buffer_size)
        self.start()

    def set_video(self, src=None, cap=None, is_loop=False):
        self.inner_lock.acquire()
        if cap is not None:
            self.cap = cap
        else:
            self.cap = cv2.VideoCapture(src) # jiarun: 从路径src读取视频
        self.frame_cnt = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) # jiarun: 返回这个视频的帧总数
        self.is_loop = is_loop
        self.inner_lock.release()

    def update(self):
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                break
            self.inner_lock.acquire()
            ret, frame = self.cap.read()
            if not ret:
                if self.is_loop and self.frame_cnt:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    frame = None
            self.inner_lock.release()
            self.queue.put(frame)
        self.cap.release()

    def read(self, block=True):
        try:
            return self.queue.get(block=block)
        except:
            return None

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.start()
