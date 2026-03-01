import time, base64
# pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

ENDPOINT = "https://otlp-gateway-prod-ap-southeast-1.grafana.net/otlp/v1/metrics"
USER = "REMOVED"
TOKEN = "REMOVED"
SERVICE = "aidetect-srcgen"


auth = "Basic " + base64.b64encode(f"{USER}:{TOKEN}".encode()).decode()
resource = Resource(attributes={
    SERVICE_NAME: SERVICE,
})
reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=ENDPOINT, headers={"Authorization": auth}),
    export_interval_millis=5000 
)
provider = MeterProvider(metric_readers=[reader], resource=resource)

# ======

meter = provider.get_meter("meter")
tokens_counter = meter.create_counter("llm_tokens")
chars_counter = meter.create_counter("llm_output_chars")


counter = meter.create_counter("test_counter")

if __name__ == "__main__":
    while True:
        time.sleep(1)
        counter.add(1, {"host": "epyc"})
        print("Counter +1")
