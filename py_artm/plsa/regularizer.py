from ..utils import call_ignore_extra_args


class RegularizerBase(object):

    def __init__(self):
        pass

    def _dr_dphi(self):
        return 0

    def dr_dphi(self, plsa):
        return call_ignore_extra_args(self._dr_dphi, plsa)

    def _dr_dtheta(self):
        return 0

    def dr_dtheta(self, plsa):
        return call_ignore_extra_args(self._dr_dtheta, plsa)


class RegularizerCoefficientBase(object):

    def __init__(self):
        pass

    def _coefficient(self):
        return 0

    def coefficient(self, plsa):
        return call_ignore_extra_args(self._coefficient, plsa)

    def __mul__(self, other):
        if isinstance(other, RegularizerBase):
            return RegularizerWithCoefficient(other, self)
        raise NotImplemented


class RegularizerWithCoefficient(object):

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


class RegularizersCombination(object):

    def __init__(self, *regularizers):
        self.regularizers = regularizers

    def dr_dphi(self, plsa):
        return sum(reg.dr_dphi(plsa) for reg in self.regularizers)

    def dr_dtheta(self, plsa):
        return sum(reg.dr_dtheta(plsa) for reg in self.regularizers)

    def __add__(self, other):
        if isinstance(other, RegularizerWithCoefficient):
            return RegularizersCombination(*(self.regularizers + (other,)))
        raise NotImplemented

    __radd__ = __add__
