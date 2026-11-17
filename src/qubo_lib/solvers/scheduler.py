import multiprocessing
import time
import queue

def devices_scheduler(devices, tasks, process_task_func, timeout=None):
    ctx = multiprocessing.get_context('spawn')

    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for task_id, (task_args, task_kwargs) in tasks.items():
        task_queue.put((task_id, task_args, task_kwargs))

    num_workers = len(devices)
    for _ in range(num_workers):
        task_queue.put(None)

    processes = []
    for worker_id, worker_device in enumerate(devices):
        args = (worker_id, worker_device, task_queue, result_queue, process_task_func)
        p = ctx.Process(target=worker, args=args)
        processes.append(p)
        p.start()

    results = {}
    actual_timeout = timeout if timeout is not None else 24 * 3600
    deadline = time.time() + actual_timeout 

    num_tasks = len(tasks)
    for _ in range(num_tasks):
        try:
            remaining = max(0, deadline - time.time())
            if remaining <= 0:
                break
            worker_id, task_id, result = result_queue.get(timeout=remaining)
            results[task_id] = (worker_id, result)
        except queue.Empty:
            break

    for p in processes:
        if p.is_alive():
            p.terminate()
        p.join()

    return results


def worker(worker_id, worker_device, task_queue, result_queue, process_task_func):
    while True:
        item = task_queue.get()
        if item is None:
            break
        task_id, task_args, task_kwargs = item

        try:
            result = process_task_func(worker_device, *task_args, **task_kwargs)
            result_queue.put((worker_id, task_id, result))
        except Exception as e:
            result_queue.put((worker_id, task_id, e))