import random


def polarity(region, gx):
    """:argument
    compute the polarity of a line segment.
    positive or negarive
    input:  region  --- line segment rect
            gx      --- horizontal gradient
    """
    s = 0
    for i in range(5):
        k = random.randint(0, len(region[0, :]) - 1)
        r = region[0, k]
        c = region[1, k]
        s = s + gx[r, c]
    if s > 0:
        p = 1
    else:
        p = -1
    return p
