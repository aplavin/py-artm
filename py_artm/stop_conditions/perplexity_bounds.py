from ..plsa import StopConditionBase


class PerplexityBounds(StopConditionBase):

    def __init__(self, lo=float('-inf'), hi=float('inf')):
        self.lo = lo
        self.hi = hi

    def _is_stop(self, perplexity):
        return not (self.lo <= perplexity <= self.hi)
