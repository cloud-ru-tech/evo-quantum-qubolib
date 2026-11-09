import time
import numpy as np

class PerformanceTimer:
    def __init__(self, use_gpu=False):
        self.times = []
        self.cp = None

        self.use_gpu = use_gpu
        if self.use_gpu:
            import cupy as cp
            self.cp = cp
            self.start_event = self.cp.cuda.Event()
            self.start_event.record()
            self.last_event = self.start_event
        else:
            self.start_time = time.perf_counter()

    def pick_time(self, label=None):
        if self.use_gpu:
            current_event = self.cp.cuda.Event()
            current_event.record()
            current_event.synchronize()

            elapsed_time_msec = self.cp.cuda.get_elapsed_time(self.last_event, current_event) # в миллисекундах
            self.last_event = current_event
        else:
            current_time = time.perf_counter()
            elapsed_time_msec = (current_time - self.start_time) * 1000 # в миллисекундах
            self.start_time = current_time

        self.times.append((elapsed_time_msec, label))

    def get_time_from_start(self):
        rez = 0
        for t, label in self.times:
            rez += t
        return rez

