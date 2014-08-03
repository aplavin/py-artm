from ..plsa import RegularizerCoefficientBase


class Constant(RegularizerCoefficientBase):

    def __init__(self, value):
        self.value = value

    def _coefficient(self):
        return self.value


class ZeroThenConstant(RegularizerCoefficientBase):

    def __init__(self, steps_zero, value):
    	self.steps_zero = steps_zero
        self.value = value

    def _coefficient(self, itnum):
        return self.value if itnum >= self.steps_zero else 0
