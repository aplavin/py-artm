from ..plsa import RegularizerCoefficientBase
from ..utils import public


@public
class ZeroThenConstant(RegularizerCoefficientBase):

    def __init__(self, steps_zero, value):
    	self.steps_zero = steps_zero
        self.value = value

    def _coefficient(self, itnum):
        return self.value if itnum >= self.steps_zero else 0
