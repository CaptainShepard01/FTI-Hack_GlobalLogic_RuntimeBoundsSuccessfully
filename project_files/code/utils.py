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
    a1, a2 = 0, 0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list) - 1):
        a1 += x_list[j] * y_list[j + 1]
        a2 += y_list[j] * x_list[j + 1]
    l = abs(a1 - a2) / 2
    return l


def centroid(x_list, y_list):
    _len = len(x_list)
    x = sum(x_list) / _len
    y = sum(y_list) / _len
    return x, y
