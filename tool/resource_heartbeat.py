#!/usr/bin/env python3
"""
Prints the machine's memory, swap and disk use every few seconds, until the shell that started it exits. A portable
counterpart to scripts/log-memory.sh, for runners without `free`.

Usage: python tool/resource_heartbeat.py <interval-seconds> [<disk-path>] &
"""

import os
import sys
import time

import psutil

GB = 1024**3


def main() -> None:
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    disk_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('RUNNER_TEMP', '.')
    parent = psutil.Process().ppid()
    while psutil.pid_exists(parent):
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage(disk_path)
        procs = []
        for p in psutil.process_iter(['name', 'memory_info']):
            mi = p.info['memory_info']
            if mi is not None:
                procs.append((mi.rss, p.info['name']))
        largest = ','.join(f'{name}:{rss / GB:.1f}G' for rss, name in sorted(procs, reverse=True)[:3])
        print(
            f'[heartbeat {time.strftime("%H:%M:%S", time.gmtime())}] '
            f'avail={vm.available / GB:.1f}G/{vm.total / GB:.1f}G swap={swap.used / GB:.1f}G '
            f'disk_free={disk.free / GB:.1f}G largest={largest}',
            flush=True,
        )
        time.sleep(interval)


if __name__ == '__main__':
    main()
