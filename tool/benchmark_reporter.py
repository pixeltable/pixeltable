import json
import logging
import os
import sys

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

GRAFANA_OTLP_URL = 'https://otlp-gateway-prod-us-west-0.grafana.net/otlp/v1/metrics'

_logger = logging.getLogger('pixeltable')


def report_benchmarks_to_grafana(json_path: str, grafana_token: str) -> None:
    with open(json_path) as f:
        data = json.load(f)

    commit_info = data.get('commit_info', {})
    branch = commit_info.get('branch', 'unknown')
    commit = commit_info.get('id', 'unknown')

    resource = Resource.create({'service.name': 'benchmark', 'environment': 'ci'})
    exporter = OTLPMetricExporter(endpoint=GRAFANA_OTLP_URL, headers={'Authorization': f'Bearer {grafana_token}'})
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    meter = metrics.get_meter('benchmark')
    mean_gauge = meter.create_gauge(
        name='benchmark_mean_seconds', description='Mean benchmark execution time', unit='s'
    )
    stddev_gauge = meter.create_gauge(
        name='benchmark_stddev_seconds', description='Benchmark execution time standard deviation', unit='s'
    )

    for bench in data.get('benchmarks', []):
        attrs = {
            'test_name': bench['name'],
            'group': bench.get('group', 'ungrouped'),
            'branch': branch,
            'commit': commit,
        }
        stats = bench['stats']
        mean_gauge.set(stats['mean'], attributes=attrs)
        stddev_gauge.set(stats['stddev'], attributes=attrs)
        _logger.info(f'Reported {bench["name"]}: mean={stats["mean"]:.4f}s stddev={stats["stddev"]:.4f}s')

    provider.shutdown()


def main() -> None:
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <benchmark-results.json>')
        sys.exit(1)

    json_path = sys.argv[1]

    grafana_token = os.getenv('GRAFANA_SERVICE_ACCOUNT_TOKEN', '')
    if not grafana_token:
        raise ValueError('GRAFANA_SERVICE_ACCOUNT_TOKEN environment variable is not set')

    logging.basicConfig(level=logging.INFO)
    report_benchmarks_to_grafana(json_path, grafana_token)


if __name__ == '__main__':
    main()
