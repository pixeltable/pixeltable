import subprocess
import sys
import time

import pixeltable as pxt


def run_workers(num_workers: int, duration: float, script: str) -> None:
    processes: list[subprocess.Popen] = []

    for i in range(num_workers):
        p = subprocess.Popen(['python', script, str(i)])
        processes.append(p)

    time.sleep(duration)

    success = True

    for i, p in enumerate(processes):
        if p.returncode is not None and p.returncode != 0:
            success = False
            print(f'Worker {i} exited abnormally (exit code {p.returncode}).')

    for p in processes:
        p.kill()

    time.sleep(2.0)

    if not success:
        sys.exit(1)


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
