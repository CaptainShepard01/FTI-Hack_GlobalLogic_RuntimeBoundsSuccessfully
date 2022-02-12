from time import time


# Decorator to detect how much time function works
def performance(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'It took {t2 - t1} s')
        return result

    return wrapper


def divide_coordinates(xy_list):
    x_list = []
    y_list = []
    for i in range(len(xy_list)):
        x_list.append(xy_list[i][0][0])
        y_list.append(xy_list[i][0][1])
    return x_list, y_list


def calculate_area(x_list, y_list):
    a = 0.0
    n = len(x_list)
    j = n - 1
    for i in range(n):
        a += (x_list[j] + x_list[i]) * (y_list[j] - y_list[i])
        j = i
    a = abs(a) / 2
    return a
