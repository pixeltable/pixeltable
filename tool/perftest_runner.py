import base64
import logging
import os

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

import pixeltable as pxt
import pixeltable.functions as pxtf
import tool.perftest_providers as ptp

GRAFANA_OTLP_URL = 'https://otlp-gateway-prod-us-west-0.grafana.net/otlp/v1/metrics'

_logger = logging.getLogger('pixeltable')


def main() -> None:
    in_ci = os.environ.get('GITHUB_ACTIONS', 'false').lower() == 'true'
    _logger.info(f'Running in CI: {in_ci}')
    if in_ci:
        grafana_api_key = os.getenv('GRAFANA_API_KEY', '')
        if not grafana_api_key:
            raise ValueError('GRAFANA_API_KEY environment variable is not set')
        grafana_instance_id = os.getenv('GRAFANA_INSTANCE_ID', '')
        if not grafana_instance_id:
            raise ValueError('GRAFANA_INSTANCE_ID environment variable is not set')

    n = 5000
    t = 1000
    provider_config = ptp.ProviderConfig(
        prompt_udf=ptp.create_chatgpt_prompt,
        udf=pxtf.openai.chat_completions,
        default_model='gpt-4o-mini',
        kwargs={'model_kwargs': {'max_tokens': t, 'temperature': 0.7}},
    )
    pxt.configure_logging(to_stdout=True, level=logging.DEBUG)
    duration, num_exc = ptp.execute_perf_test(n=n, t=t, provider=provider_config, recompute_excs=True)
    success = num_exc == 0
    throughput = n / duration.total_seconds() if duration.total_seconds() > 0 else None

    throughput_str = f'{throughput:.2f}' if throughput is not None else None
    _logger.info(
        f'(n={n}, t={t}) perf test completed in {duration}, throughput={throughput_str}, exceptions={num_exc},'
        f' success={success}. Provider config was: {provider_config}'
    )
    if in_ci:
        report_to_grafana('openai', success, throughput, grafana_instance_id, grafana_api_key)

    if not success:
        raise RuntimeError('Performance test failed')


def report_to_grafana(
    test_name: str, success: bool, throughput: float | None, grafana_instance_id: str, grafana_api_token: str
) -> None:
    resource = Resource.create({'service.name': 'perf-test', 'environment': 'ci'})
    auth = base64.b64encode(f'{grafana_instance_id}:{grafana_api_token}'.encode()).decode()
    exporter = OTLPMetricExporter(endpoint=GRAFANA_OTLP_URL, headers={'Authorization': f'Basic {auth}'})
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    meter = metrics.get_meter('performance')
    status_counter = meter.create_up_down_counter(
        name='perf_test_status_tmp', description='CI test execution status (1=success, 0=failure)', unit='1'
    )

    status_counter.add(int(success), attributes={'test_name': test_name, 'status': 'success' if success else 'failure'})

    if throughput is not None:
        throughput_gauge = meter.create_gauge(
            name='perf_test_throughput_tmp', description='Request throughput in requests per second', unit='req/s'
        )

        throughput_gauge.set(throughput, attributes={'test_name': test_name})

    provider.shutdown()


if __name__ == '__main__':
    main()
