from ..utils import call_ignore_extra_args, public


@public
class QuantityBase(object):

    def __init__(self):
        pass

    def _items(self):
        return []

    items = call_ignore_extra_args(_items)
