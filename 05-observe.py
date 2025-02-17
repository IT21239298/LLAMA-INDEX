import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core import Settings
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # Changed to HTTP exporter
from opentelemetry import trace
import logging

# Configure logging to suppress warnings
logging.basicConfig(level=logging.ERROR)
os.environ["GRPC_PYTHON_LOG_LEVEL"] = "error"

load_dotenv()

# Phoenix setup
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")

# Configure OpenTelemetry with Phoenix using HTTP instead of gRPC
tracer_provider = TracerProvider()
phoenix_exporter = OTLPSpanExporter(
    endpoint="https://ingest.phoenix.arize.com/v1/traces",
    headers={
        "authorization": f"Bearer {PHOENIX_API_KEY}",
    },
)
span_processor = BatchSpanProcessor(phoenix_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)

# OpenAI setup
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

try:
    documents = SimpleDirectoryReader("pdf/").load_data()

    if os.path.exists("storage"):
        print("Loading index from storage")
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        index = load_index_from_storage(storage_context)
    else:
        print("Creating new index")
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir="storage")

    query_engine = index.as_query_engine()
    response = query_engine.query("What are the design goals and give details about it please.")
    print(response)

finally:
    # Clean shutdown
    span_processor.shutdown()
    phoenix_exporter.shutdown()