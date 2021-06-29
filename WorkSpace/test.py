from concurrent.futures import ThreadPoolExecutor
import threading

# cdef int i

def print_index(i):
    print()


with ThreadPoolExecutor(max_workers=8) as e:
    for i in range(10000):
        e.submit(print_index, i)

print('hello')