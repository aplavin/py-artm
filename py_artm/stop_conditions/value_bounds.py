from functools import partial
from ..plsa import StopConditionBase
from ..utils import public


@public
class ValueBounds(StopConditionBase):

    def __init__(self, value_name, lo=float('-inf'), hi=float('inf')):
        exec "self._is_stop = (lambda lo=lo, hi=hi: lambda %(value_name)s: not (lo <= %(value_name)s <= hi))()" % locals()
