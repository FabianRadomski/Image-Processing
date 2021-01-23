from multiprocessing import JoinableQueue, cpu_count
from threading import Thread
from random import randrange
from time import time

q = JoinableQueue(maxsize=0)
single_q = []
num_threads = cpu_count() -
print('Num of threads ' + str(num_threads))

for i in range(1000):
    num = randrange(9999,99999)
    q.put([i, num])
    single_q.append(num)


results = []
results_single = []

def calc(num):
    for _ in range(num):
        2**21
    return num

def process(queue, result):
    while not q.empty():
        work = q.get()
        result.append(calc(work[1]))
        q.task_done()
    return True


# single_time_start = time()
# for i in single_q:
#     results_single.append(calc(i))
# single_time_end = time()
# print('Time single thread: ' + str((single_time_end - single_time_start) * 1000))

multi_time_start = time()
for i in range(num_threads):
    worker = Thread(target=process, args=(q,results))
    worker.setDaemon(True)
    worker.start()
multi_time_end = time()

q.join()
print('Time multi thread: ' + str((multi_time_end - multi_time_start) * 1000))
