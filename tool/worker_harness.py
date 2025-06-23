import subprocess
import sys
import time
from io import TextIOWrapper

import pixeltable as pxt


def run_workers(num_workers: int, duration: float, script: str) -> None:
    processes: list[subprocess.Popen] = []
    log_files: list[TextIOWrapper] = []

    for i in range(num_workers):
        log_file = open(f'worker-{i}.log', 'w')
        p = subprocess.Popen(['python', script, str(i)])
        processes.append(p)
        log_files.append(log_file)

    time.sleep(duration)

    for p in processes:
        p.kill()

    time.sleep(2.0)

    for log_file in log_files:
        log_file.close()


def main() -> None:
    if len(sys.argv) != 4:
        print(f'Usage: python {sys.argv[0]} <num-workers> <duration-secs> <script.py>')
        sys.exit(1)
    try:
        num_workers = int(sys.argv[1])
        duration = float(sys.argv[2])
    except ValueError:
        print(f'Usage: python {sys.argv[0]} <num-workers> <duration-secs> <script.py>')
        sys.exit(1)
    script = sys.argv[3]

    pxt.init()
    run_workers(num_workers, duration, script)


if __name__ == '__main__':
    main()
