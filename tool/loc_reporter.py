import base64
import logging
import os
import subprocess
import sys
from collections import defaultdict

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

GRAFANA_OTLP_URL = 'https://otlp-gateway-prod-us-west-0.grafana.net/otlp/v1/metrics'

_logger = logging.getLogger('pixeltable')


def count_lines_by_author(repo_path: str) -> dict[str, int]:
    """Returns total lines added per author across all commits (excluding merges)."""
    result = subprocess.run(
        ['git', '-C', repo_path, 'log', '--numstat', '--pretty=%aN', '--no-merges'],
        capture_output=True,
        text=True,
        check=True,
    )

    lines_by_author: dict[str, int] = defaultdict(int)
    current_author: str | None = None

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 3:
            added, deleted, _ = parts
            if added != '-' and current_author is not None:
                lines_by_author[current_author] += int(added)
        else:
            current_author = line

    return dict(lines_by_author)


def report_loc_to_grafana(repo_path: str, grafana_instance_id: str, grafana_token: str) -> None:
    lines_by_author = count_lines_by_author(repo_path)

    resource = Resource.create({'service.name': 'repository', 'environment': 'ci'})
    auth = base64.b64encode(f'{grafana_instance_id}:{grafana_token}'.encode()).decode()
    exporter = OTLPMetricExporter(endpoint=GRAFANA_OTLP_URL, headers={'Authorization': f'Basic {auth}'})
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    meter = metrics.get_meter('loc')
    loc_gauge = meter.create_gauge(
        name='repository_loc_by_author', description='Total lines of code added per author', unit='1'
    )

    for author, lines in lines_by_author.items():
        loc_gauge.set(lines, attributes={'author': author})
        _logger.info(f'{author}: {lines} lines')

    provider.shutdown()


def main() -> None:
    repo_path = sys.argv[1] if len(sys.argv) > 1 else '.'

    grafana_instance_id = os.getenv('GRAFANA_INSTANCE_ID', '')
    if not grafana_instance_id:
        raise ValueError('GRAFANA_INSTANCE_ID environment variable is not set')
    grafana_token = os.getenv('GRAFANA_SERVICE_ACCOUNT_TOKEN', '')
    if not grafana_token:
        raise ValueError('GRAFANA_SERVICE_ACCOUNT_TOKEN environment variable is not set')

    logging.basicConfig(level=logging.INFO)
    report_loc_to_grafana(repo_path, grafana_instance_id, grafana_token)


if __name__ == '__main__':
    main()
