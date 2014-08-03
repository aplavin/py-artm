from ..utils import call_ignore_extra_args


class QuantityBase(object):

    def __init__(self):
        pass

    def _items(self):
        return []

    def items(self, plsa):
        return call_ignore_extra_args(self._items, plsa)
