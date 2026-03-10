import subprocess
import sys
import time


def _terminate_workers(processes: list[subprocess.Popen], start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f'Terminating all workers after {elapsed:.1f} seconds.')
    for p in processes:
        if p.poll() is None:
            p.kill()
    time.sleep(2.0)


def run_workers(
    num_workers: int, duration: float, script: str | None = None, worker_args: list[list[str]] | None = None
) -> None:
    if (script is None) == (worker_args is None):
        raise ValueError('Exactly one of `script` or `worker_args` must be specified.')

    if script is not None:
        worker_args = [[script, str(i)] for i in range(num_workers)]

    processes: list[subprocess.Popen] = []
    for i in range(num_workers):
        p = subprocess.Popen(['python', *worker_args[i]])
        processes.append(p)

    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            # Check if any process has terminated
            for i, p in enumerate(processes):
                returncode = p.poll()
                if returncode is not None:
                    is_error = returncode != 0
                    print(f'Worker {i} exited {"with error" if is_error else ""} (exit code {returncode}).')
                    _terminate_workers(processes, start_time)
                    sys.exit(returncode)

            time.sleep(0.1)  # Small delay to avoid busy waiting

    except KeyboardInterrupt:
        print('\n\nReceived KeyboardInterrupt.')
        _terminate_workers(processes, start_time)
        sys.exit(130)

    # Duration elapsed; kill all processes
    _terminate_workers(processes, start_time)


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
    run_workers(num_workers, duration, script)


if __name__ == '__main__':
    main()
