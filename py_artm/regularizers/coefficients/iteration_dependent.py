from ...plsa import RegularizerCoefficientBase
from ...utils import public


@public
class ZeroThenConstant(RegularizerCoefficientBase):

    def __init__(self, steps_zero, value):
    	self.steps_zero = steps_zero
        self.value = value

    def _coefficient(self, itnum):
        return self.value if itnum >= self.steps_zero else 0


@public
class Linear(RegularizerCoefficientBase):

    def __init__(self, max_value):
        self.max_value = max_value

    def _coefficient(self, itnum, maxiter):
        return self.max_value * 1.0 * itnum / maxiter
