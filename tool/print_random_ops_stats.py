import re
from typing import ClassVar

import pandas as pd

from pixeltable.config import Config
from tool.random_tbl_ops_2 import RandomTblOps

LINE_PARSER = re.compile(
    r'^\[(?P<timestamp>.+?)\] \[Worker (?P<worker_id>\d+)\] \[(?P<operation>.+?)\]: (?P<message>.*)$'
)


class StatsPrinter:
    num_operations: ClassVar[dict[str, dict[int, int]]] = {}
    num_errors: ClassVar[dict[str, dict[int, int]]] = {}
    error_freq: ClassVar[dict[str, dict[int, int]]] = {}
    max_worker: ClassVar[int] = 0

    def process_line(self, line: str) -> None:
        m = LINE_PARSER.match(line.strip())
        if m is None:
            return

        worker = int(m['worker_id'])
        self.max_worker = max(self.max_worker, worker)
        operation = m['operation'].strip()
        message = m['message']

        if operation not in self.num_operations:
            self.num_operations[operation] = {}
        op_dict = self.num_operations[operation]
        if worker not in op_dict:
            op_dict[worker] = 0
        op_dict[worker] += 1

        if 'pxt.Error' in message:
            if operation not in self.num_errors:
                self.num_errors[operation] = {}
            err_dict = self.num_errors[operation]
            if worker not in err_dict:
                err_dict[worker] = 0
            err_dict[worker] += 1

            idx = message.index('pxt.Error')
            message = message[idx:].removeprefix('pxt.Error: ')
            err_note = message[:30]
            if err_note not in self.error_freq:
                self.error_freq[err_note] = {}
            err_dict = self.error_freq[err_note]
            if worker not in err_dict:
                err_dict[worker] = 0
            err_dict[worker] += 1

    def print_stats(self) -> None:
        with open(Config.get().home / 'random-tbl-ops.log', encoding='utf-8') as fp:
            while True:
                line = fp.readline()
                if line == '':
                    break
                self.process_line(line)

        ops = tuple(op for op, *_ in RandomTblOps.RANDOM_OPS_DEF)
        data: dict[str, list[str]] = {}
        for op in ops:
            key = op[:6]
            num_ops = sum(self.num_operations.get(op, {}).values())
            num_errs = sum(self.num_errors.get(op, {}).values())
            err_pct = (100.0 * num_errs / num_ops) if num_ops > 0 else 0.0
            data[key] = [f'{num_ops} ', f'{num_errs} ', f'{err_pct:.0f}%']

        print(pd.DataFrame(data, index=['Operations', 'Errors', 'Error %']))
        print()

        err_notes = sorted(self.error_freq.keys(), key=lambda k: -sum(self.error_freq[k].values()))
        err_data: dict[str, list[int]] = {}
        err_data['Total'] = [sum(self.error_freq[err_note].values()) for err_note in err_notes]
        for worker in range(self.max_worker + 1):
            err_data[f'W {worker:02d}'] = [self.error_freq[err_note].get(worker, 0) for err_note in err_notes]

        print(pd.DataFrame(err_data, index=err_notes))


def main() -> None:
    StatsPrinter().print_stats()


if __name__ == '__main__':
    main()
