import inspect
import os
import psutil
import sys
import time

def elapsed_since(start):
    elapsed = time.time() - start
    if elapsed < 1:
        return str(round(elapsed*1000,2)) + "ms"
    if elapsed < 60:
        return str(round(elapsed, 2)) + "s"
    if elapsed < 3600:
        return str(round(elapsed/60, 2)) + "min"
    else:
        return str(round(elapsed/3600, 2)) + "h"

def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_full_info()
    return mem_info.rss, mem_info.vms, mem_info.shared, mem_info.uss

def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes) + "B"
    if abs(bytes) < 1e6:
        return str(round(bytes/1e3,2)) + "kB"
    if abs(bytes) < 1e9:
        return str(round(bytes/1e6,2)) + "MB"
    else:
        return str(round(bytes/1e9,2)) + "GB"

def memory_snapshot(obj, label=None):
    rss, vms, shared, uss = get_process_memory()
    if label is not None:
        label_string = f"<{label}>"
    print("RSS: {:>8} | VMS: {:>8} | SHR {" ":>8} | USS: {:>8}"
              .format(format_bytes(rss),
                      format_bytes(vms),
                      format_bytes(shared),
                      format_bytes(uss)))
    contents = vars(obj)
    sorted_contents = sorted([(var, sys.getsizeof(contents[var]))
                              for var in contents],
                             key=lambda tuple:tuple[1],
                             reverse=True)
    for var, size in sorted_contents:
        print("{}\t\t{}"
              .format(format_bytes(size),
                      var))
                    
def profile(func):
    def wrapper(*args, **kwargs):
        rss_before, vms_before, shared_before, uss_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        rss_after, vms_after, shared_after, uss_after = get_process_memory()
        print("{:>20} RSS: {:>8} | VMS: {:>8} | SHR {" ":>8} | USS: {:>8} | "
              "time: {:>8}"
              .format("<" + func.__name__ + ">",
                      format_bytes(rss_after - rss_before),
                      format_bytes(vms_after - vms_before),
                      format_bytes(uss_after - uss_before),
                      format_bytes(shared_after - shared_before),
                      elapsed_time))
        return result
    if inspect.isfunction(func):
        return wrapper
    if inspect.ismethod(func):
        return wrapper(*args, **kwargs)

def show_stack():
    stack = inspect.stack()[1:]
    for frame in stack:
        print(f"Line {frame.lineno}, {frame.filename}")

