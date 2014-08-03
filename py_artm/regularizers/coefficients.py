from ..plsa import RegularizerCoefficientBase


class Constant(RegularizerCoefficientBase):

    def __init__(self, value):
        self.value = value

    def _coefficient(self):
        return self.value
