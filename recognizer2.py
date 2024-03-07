from speech_recognition import Recognizer, Microphone, WaitTimeoutError
from conf import conf, load_setting
# import win32com.client
# from win32com.client import constants
from comtypes.client import CreateObject, Constants


# 添加百度识别接口
class ExtentRecognizer(Recognizer):
    def __init__(self):
        super(ExtentRecognizer, self).__init__()

    def recognize_baidu(self, audio_data, auth_info):
        from aip import AipSpeech
        app_id = auth_info['APP_ID']
        app_key = auth_info['API_KEY']
        secret_key = auth_info['SECRET_KEY']
        client = AipSpeech(app_id, app_key, secret_key)

        # 采样率1.6kHz, pcm格式, 位宽16bit, 1537: 汉语普通话, 1936: 远场识别
        pcm_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
        ret = client.asr(pcm_data, 'pcm', 16000, {'dev_pid': 1537})
        if not ret['err_no']:
            res = ' '.join(ret['result'])       # 将所有候选结果拼接，空格隔开
            print(res)
            return 0, res
        elif ret['err_no'] in (3302, 3304, 3305):
            # 由于权限问题引起的错误
            return 1, ret['err_msg']
        else:
            return 2, ret['err_msg']


_recognizer = ExtentRecognizer()
# _speaker = win32com.client.Dispatch("SAPI.SpVoice")
_speaker = CreateObject("SAPI.SpVoice")
constants = Constants(_speaker)


def input(phraselist, stop_event=None, speaker=None):
    with Microphone() as source:
        while True:
            if (stop_event is not None) and stop_event.is_set():
                return None
            if (speaker is not None) and speaker.Status.RunningState == 2:
                # 若正在进行语音播放，则不识别
                # ref: https://docs.microsoft.com/en-us/previous-versions/windows/desktop/ms720850(v=vs.85)
                continue
            try:
                audio = _recognizer.listen(source, timeout=0.1, phrase_time_limit=5)
            except WaitTimeoutError:
                continue
            try:
                status, ret = _recognizer.recognize_baidu(audio, conf.BAIDU_AUTH)
            except:
                continue
            if status == 0:
                for phrase in phraselist:
                    if phrase in ret:
                        return phrase
            elif status == 1:
                _speaker.Speak('百度语音识别API权限异常 请检查错误信息', constants.SVSFlagsAsync | constants.SVSFPurgeBeforeSpeak)
                print(ret)
                return None
            else:
                continue


def set_energy_thres(thres, dynamic=False):
    # 设置语音识别能量阈值
    _recognizer.dynamic_energy_threshold = dynamic
    _recognizer.energy_threshold = thres


if __name__ == '__main__':
    load_setting()
    cmd = input(phraselist=['我已完成', '回到前面'], stop_event=None)
