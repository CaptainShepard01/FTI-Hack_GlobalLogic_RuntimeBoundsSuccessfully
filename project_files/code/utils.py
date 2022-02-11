from time import time


def performance(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'It took {t2 - t1} s')
        return result

    return wrapper


def explode_xy(xy):
    xl = []
    yl = []
    for i in range(len(xy)):
        xl.append(xy[i][0][0])
        yl.append(xy[i][0][1])
    return xl, yl


def shoelace_area(x_list, y_list):
    # a = 0.0
    # n = len(x_list)
    # for i in range(n):
    #     j = (i + 1) % n
    #     a += x_list[i] * y_list[j]
    #     a -= x_list[j] * y_list[i]
    # a = abs(a) / 2
    a = 0.0
    n = len(x_list)
    j = n - 1
    for i in range(n):
        a += (x_list[j] + x_list[i]) * (y_list[j] - y_list[i])
        j = i
    a = abs(a) / 2
    return a
