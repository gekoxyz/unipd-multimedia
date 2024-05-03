import math


def expgolomb_signed(n):
    if n > 0:
        return expgolomb_unsigned(2 * n - 1)
    else:
        return expgolomb_unsigned(-2 * n)


def expgolomb_unsigned(n):
    if n == 0:
        return 1
    else:
        return (2 * math.ceil(math.log2(n + 2))) - 1
