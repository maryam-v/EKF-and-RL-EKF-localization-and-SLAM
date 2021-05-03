import math


def con_bear(old_bear):
    while old_bear < -math.pi:
        old_bear = old_bear + 2 * math.pi
    while old_bear > math.pi:
        old_bear = old_bear - 2 * math.pi
    new_bear = old_bear
    return new_bear