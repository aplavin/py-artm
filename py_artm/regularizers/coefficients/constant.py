from ...plsa import RegularizerCoefficientBase
from ...utils import public


@public
class Constant(RegularizerCoefficientBase):

    def __init__(self, value):
        self.value = value

    def _coefficient(self):
        return self.value
