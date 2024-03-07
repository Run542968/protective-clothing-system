from comtypes.client import CreateObject, Constants


class Speaker:
    def __init__(self):
        self.normal_speaker = CreateObject("SAPI.SpVoice")
        self.constants = Constants(self.normal_speaker)

        self.prior_speaker = CreateObject("SAPI.SpVoice")
        self.prior_speaker.Priority = self.constants.SVPAlert

    def speak(self, msg, type_, purge=True):
        if type_ == 'NORMAL':
            speaker = self.normal_speaker
        elif type_ == 'PRIOR':
            speaker = self.prior_speaker
        else:
            raise Exception('Unknown Type')

        msgs = msg.split()
        if purge:
            speaker.Speak(msgs[0], self.constants.SVSFlagsAsync | self.constants.SVSFPurgeBeforeSpeak)
        else:
            speaker.Speak(msgs[0], self.constants.SVSFlagsAsync)
        for m in msgs[1:]:
            speaker.Speak(m, self.constants.SVSFlagsAsync)
