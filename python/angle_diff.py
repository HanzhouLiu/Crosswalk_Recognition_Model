import math


def angle_diff(t1, t2):
    """:cvar
        input: two angles t1 and t2 in [-pi, pi]
        output: smallest difference to rotate one to match the other
    """
    diff = abs(t1 - t2)
    if diff > math.pi:
        diff = 2*math.pi - diff
    return diff
