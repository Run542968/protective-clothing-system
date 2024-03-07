import os
from datetime import datetime
from subprocess import Popen, PIPE
from utils import _clean_dir


class VideoWriter:
    def __init__(self, path, resolution, fps=25, storage_limit=0):
        if not os.path.exists(path):
            os.makedirs(path)
        dt = datetime.now()
        file_name = dt.strftime('%Y_%m_%d_%H_%M_%S') + '.avi'
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', resolution,        # e.g. '720x1280'
                   '-pix_fmt', 'bgr24',
                   '-r', str(fps),
                   '-i', '-',
                   '-an',
                   '-vcodec', 'mpeg4',
                   '-b:v', '5000k',
                   '-t', '1800',
                   os.path.join(path, file_name)]
        self.video_storage = Popen(command, stdin=PIPE)
        if storage_limit > 0:
            _clean_dir(path, storage_limit * (1024 ** 3))

    def __del__(self):
        self.video_storage.stdin.close()
        self.video_storage.wait()

    def write(self, cv2img):
        self.video_storage.stdin.write(cv2img.tostring())
