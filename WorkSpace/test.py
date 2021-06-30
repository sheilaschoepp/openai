from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import Process, Pool, Value
import time
import threading
from itertools import product
import os


# cdef int i

def f(msg, num=1):
    print(mp.current_process())
    print(msg + ' ' + str(num))


def wait(sec, msg):
    # print(mp.current_process())
    print(f'sleeping for {sec} seconds {msg}')
    time.sleep(sec)
    print(f'done sleeping {sec} seconds')

with ThreadPoolExecutor() as executor:
    secs = [(5, 'h1'), (4, 'h2'), (3, 'h3'), (2, 'h4'), (1, 'h5')]
    [executor.submit(wait, sec, msg) for sec, msg in secs]

# p1 = Process(target=wait)
# p2 = Process(target=wait())
#
# p1.join()
# p2.join()

# with ThreadPoolExecutor(max_workers=8) as e:
#     for i in range(10000):
#         e.submit(print_index, i)
