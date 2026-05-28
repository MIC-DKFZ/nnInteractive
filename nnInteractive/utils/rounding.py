from math import ceil, floor


def round_to_nearest_odd(number: float):
    assert number > 0
    cl = ceil(number)
    fl = floor(number)
    if cl % 2 == 1:
        return cl
    elif fl % 2 == 1:
        return fl
    else:
        return round(number) + 1
