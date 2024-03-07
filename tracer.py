from queue import Queue
from conf import conf


class Tracer:
    def __init__(self):
        self.tracer = None
        self.load_tracer(0)

    def load_tracer(self, cur_idx):
        if conf.STEPS[cur_idx].split()[1] == "手卫生":
            self.tracer = ParallelTracer(list(range(1, len(conf.HWD_STEPS))),
                                         conf.HW_WINDOW_SIZE, conf.HW_ACTIVATE_THRES)
        else:
            self.tracer = TrivialTracer(conf.WINDOW_SIZE[cur_idx], conf.ACTIVATE_THRES[cur_idx])
        # self.tracer = TrivialTracer(conf.WINDOW_SIZE[cur_idx], conf.ACTIVATE_THRES[cur_idx])

    def trace(self, predict):
        return self.tracer.trace(predict)


class TrivialTracer:
    def __init__(self, max_len, activate_thres):
        self.window = Queue()
        self.activate_count = 0

        self.max_len = max_len
        self.activate_thres = activate_thres

    def trace(self, predict, verbose=True):
        if self.window.qsize() == self.max_len:
            self.activate_count -= self.window.get()
        self.activate_count += int(predict > 0)
        self.window.put(int(predict > 0))
        if verbose:
            print(self.window.qsize(), self.activate_count)

        if self.activate_count >= int(self.max_len * self.activate_thres):
            return True
        else:
            return False

    def reset(self):
        self.window = Queue()
        self.activate_count = 0


# 可以同时追踪多个类别
class ParallelTracer:
    def __init__(self, class_lst, max_len, activate_thres):
        self.tracer_lst = dict(zip(class_lst, [TrivialTracer(max_len, activate_thres) for _ in class_lst]))
        self.done = set()

    def trace(self, predict, ret_type=0):
        predict = int(predict)
        this_done = -1          # 本轮完成的类
        try:
            flag = self.tracer_lst[predict].trace(1, verbose=False)
            if flag: # 这个动作完成了
                old_len = len(self.done)
                self.done.add(predict)
                if old_len < len(self.done):
                    this_done = predict
        except:
            pass
        [self.tracer_lst[p].trace(0, verbose=False) for p in self.tracer_lst.keys() if p != predict] # 非当前动作都trace(0)
        if ret_type == 0:
            return len(self.done) == len(self.tracer_lst)       # 是否跟踪完所有类别
        elif ret_type == 1:
            return this_done            # 返回本轮完成的类
        else:
            raise NotImplementedError


# class HandWashTracer:
#     def __init__(self):
#         self.pipeline = [(1,), (2, 3), (4,), (5, 6), (7, 8), (9, 10), (11, 12), (-1,)]
#         self.cur_idx = 0
#         self.tracer = ParallelTracer(self.pipeline[self.cur_idx], conf.HW_WINDOW_SIZE, conf.HW_ACTIVATE_THRES)
#
#     def trace(self, predict):
#         code = 0
#         flag = self.tracer.trace(predict)
#         if len(self.pipeline[self.cur_idx]) == 1:
#             if flag == 1:
#                 self.cur_idx += 1
#                 self.tracer = ParallelTracer(self.pipeline[self.cur_idx], conf.HW_WINDOW_SIZE, conf.HW_ACTIVATE_THRES)
#                 code = 2        # current done
#         else:
#             if flag == 1:
#                 code = 1        # half done
#             elif flag == 2:
#                 self.cur_idx += 1
#                 self.tracer = ParallelTracer(self.pipeline[self.cur_idx], conf.HW_WINDOW_SIZE, conf.HW_ACTIVATE_THRES)
#                 code = 2
#         if self.pipeline[self.cur_idx][0] == -1:
#             code = 3            # all done
#         return code
#
#     def reset(self):
#         self.cur_idx = 0
#         self.tracer = ParallelTracer(self.pipeline[self.cur_idx], conf.HW_WINDOW_SIZE, conf.HW_ACTIVATE_THRES)
#
#     def check(self, predict):
#         # 检测是否当前步骤
#         try:
#             return predict in self.pipeline[self.cur_idx] or predict in self.pipeline[self.cur_idx-1]
#         except:
#             return False
