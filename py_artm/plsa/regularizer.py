from ..utils import call_ignore_extra_args, public
from .quantity import QuantityBase


@public
class RegularizerBase(object):

    def __init__(self):
        pass

    def _dr_dphi(self):
        return 0

    dr_dphi = call_ignore_extra_args(_dr_dphi)

    def _dr_dtheta(self):
        return 0

    dr_dtheta = call_ignore_extra_args(_dr_dtheta)


@public
class RegularizerCoefficientBase(object):

    def __init__(self):
        pass

    def _coefficient(self):
        return 0

    coefficient = call_ignore_extra_args(_coefficient)

    def __mul__(self, other):
        if isinstance(other, RegularizerBase):
            return RegularizerWithCoefficient(other, self)
        raise NotImplemented


@public
class RegularizerWithCoefficient(QuantityBase):

    def __init__(self, regularizer, coefficient):
        self.regularizer = regularizer
        self.coefficient = coefficient

    def dr_dphi(self, plsa):
        base_val = self.regularizer.dr_dphi(plsa)
        coeff = self.coefficient.coefficient(plsa)
        return coeff * base_val

    def dr_dtheta(self, plsa):
        base_val = self.regularizer.dr_dtheta(plsa)
        coeff = self.coefficient.coefficient(plsa)
        return coeff * base_val

    def items(self, plsa):
        coeff = self.coefficient.coefficient(plsa)
        yield ('coeff_%s' % self.regularizer.__class__.__name__, coeff)
