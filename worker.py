import numpy as np


class Worker (object):

    def toBitmapMatrix (self, str):
        arr = np.zeros(784, dtype=np.uint8)
        for i in range(len(str)):
            arr[i] = int(str[i])
            if arr[i] == 1:
                arr[i] = 255
        arr = arr.reshape([28, 28])
        return arr
