from ..utils import call_ignore_extra_args


class StopConditionBase(object):

    def __init__(self):
        pass

    def _is_stop(self):
        return []

    is_stop = call_ignore_extra_args(_is_stop)
