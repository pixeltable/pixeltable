import subprocess
import sys
import time

import pixeltable as pxt


def run_workers(num_workers: int, duration: float, script: str) -> None:
    processes: list[subprocess.Popen] = []
    for i in range(num_workers):
        p = subprocess.Popen(['python', script, str(i)])
        processes.append(p)

    start_time = time.time()

    while time.time() - start_time < duration:
        # Check if any process has terminated
        for i, p in enumerate(processes):
            returncode = p.poll()
            if returncode is not None:
                end_time = time.time()
                if returncode != 0:
                    print(f'Worker {i} exited abnormally (exit code {returncode}). Terminating all workers after {end_time - start_time:.2f} seconds.')
                    # Kill all remaining processes
                    for proc in processes:
                        if proc.poll() is None:  # Still running
                            proc.kill()
                    time.sleep(2.0)
                    sys.exit(1)
                else:
                    # Process completed successfully, but we still want to stop everything
                    print(f'Worker {i} completed successfully. Terminating all workers after {end_time - start_time:.2f} seconds.')
                    for proc in processes:
                        if proc.poll() is None:  # Still running
                            proc.kill()
                    time.sleep(2.0)
                    return

        time.sleep(0.1)  # Small delay to avoid busy waiting

    # Duration elapsed, kill all processes
    print(f'Terminating all workers after {duration} seconds.')
    for p in processes:
        if p.poll() is None:  # Still running
            p.kill()
    time.sleep(2.0)


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
