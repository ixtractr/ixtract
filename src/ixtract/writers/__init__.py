"""Writers — output serialization with atomic finalize."""
from ixtract.writers.parquet import ParquetWriter, BaseWriter, WriteResult, FinalizeResult
from ixtract.writers.csv_writer import CSVWriter
from ixtract.writers.rotating import RotatingWriter

# Cloud writers are imported lazily (require boto3 / google-cloud-storage)
# from ixtract.writers.s3_writer import S3Writer
# from ixtract.writers.gcs_writer import GCSWriter

__all__ = [
    "ParquetWriter", "CSVWriter", "RotatingWriter",
    "BaseWriter", "WriteResult", "FinalizeResult",
]
