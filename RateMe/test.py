from multiprocessing import Process, Queue, cpu_count
import random
import time


def serve(queue):
    works = ["task_1", "task_2"]
    while True:
        time.sleep(0.01)
        queue.put(random.choice(works))


def work(id, queue):
    while True:
        task = queue.get()
        if task is None:
            break
        time.sleep(0.05)
        print("%d task:" % id, task)
    queue.put(None)


class Manager:
    def __init__(self):
        self.queue = Queue()
        self.NUMBER_OF_PROCESSES = cpu_count()

    def start(self):
        print("starting %d workers" % self.NUMBER_OF_PROCESSES)
        self.workers = [Process(target=work, args=(i, self.queue,))
                        for i in range(self.NUMBER_OF_PROCESSES)]
        for w in self.workers:
            w.start()

        serve(self.queue)

    def stop(self):
        self.queue.put(None)
        for i in range(self.NUMBER_OF_PROCESS):
            self.workers[i].join()
        queue.close()

Manager().start()