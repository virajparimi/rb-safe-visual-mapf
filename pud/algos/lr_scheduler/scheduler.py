import numpy as np
from typing import List


class Scheduler:
    """Scheduler template"""

    def __init__(self, start: float, stop: float):
        assert start <= stop, "start > stop"
        self.start = start
        self.stop = stop
        self.A = stop - start

    def __call__(self, inp: float):
        pass


class PiecewiseLinearSchedulerUp(Scheduler):
    def __call__(self, inp: float):
        """
        repeat: Apply symmetry and repeat
        """
        ret = 0.0
        if inp > self.stop:
            ret = 1.0
        elif inp < self.start:
            ret = 0.0
        else:
            ret = (inp - self.start) / self.A
        return ret


class PiecewiseLinearSchedulerDown(Scheduler):
    def __call__(self, inp: float):
        """
        repeat: Apply symmetry and repeat
        """
        ret = 1.0
        if inp <= self.start:
            ret = 1.0
        elif inp >= self.stop:
            ret = 0.0
        else:
            ret = 1.0 - (inp - self.start) / self.A
        return ret


class HybridScheduler(Scheduler):
    def __init__(self, start: float, stop: float, list_scheds: List[Scheduler]):
        """
        list_scheds: [sch1, sch2, ...]
        Next scheduler takes control when the passing the stop mark of the previous scheduler
        """
        super(HybridScheduler, self).__init__(start=start, stop=stop)
        # Sanity check
        assert len(list_scheds) > 0, "List of schedulers cannot be empty"
        self.schedulers = list_scheds
        self.stops = []
        for i in range(len(list_scheds)):
            self.stops.append(list_scheds[i].stop)
            if len(self.stops) > 1:
                assert self.stops[-1] >= self.stops[-2]

    def __call__(self, inp):
        for i in range(len(self.stops)):
            if inp <= self.stops[i]:
                return self.schedulers[i](inp=inp)
        return self.schedulers[-1](inp=inp)


class PiecewiseCosineScheduler:
    def __init__(self, target_margin: float, limit: float):
        self.target_margin = target_margin
        self.limit = limit
        self.A = limit - target_margin

    def __call__(self, inp: float, symmetric=True):
        ret = 0.0
        if (symmetric) and inp > self.limit + (self.limit - self.target_margin):
            ret = 0.0
        elif (not symmetric) and inp > self.limit:
            ret = 1.0
        elif inp < self.target_margin:
            ret = 0.0
        else:
            ret = (
                0.5 * np.cos(np.pi / self.A * (inp - self.A - self.target_margin)) + 0.5
            )
        return ret
