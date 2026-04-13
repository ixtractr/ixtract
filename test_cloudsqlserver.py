# run4_sqlserver.py
from ixtract import plan, execute, profile, ExtractionIntent, CostConfig
from ixtract.intent import SourceType, TargetType, ExtractionConstraints

    
intent = ExtractionIntent(
    source_type=SourceType.SQLSERVER,
    source_config={
        "connection_string": (
            "Driver={ODBC Driver 18 for SQL Server};"
            "Server=tcp:ixtract-db-server-46.database.windows.net,1433;"
            "Database=ixtract_test;"
            "Uid=ixtract;"
            "Pwd={Arky%d_rky*p_rky@1803};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            "Connection Timeout=30;"
        ),
        "database": "ixtract_test",
    },
    object_name="cloud_extraction_test",
    target_type=TargetType.PARQUET,
    target_config={"output_path": "./data/run4", "object_name": "cloud_extraction_test"},
    constraints=ExtractionConstraints(),
)

# Step 1: Profile
prof = profile(intent)
print(f"\nProfile: {prof.object_name}")
print(f"  Rows: {prof.row_estimate:,}  |  PK: {prof.primary_key}")
print(f"  Latency: p50={prof.latency_p50_ms:.1f}ms  p95={prof.latency_p95_ms:.1f}ms")

# Step 2: Plan with cost
result = plan(intent, cost_config=CostConfig(
    compute_cost_per_hour=0.50,
    egress_cost_per_gb=0.09,
))
print(f"\nPlan: {result.execution_plan.worker_count} workers")
print(f"  Verdict: {result.verdict.label}")
if result.cost_estimate:
    print(f"  Cost: ${result.cost_estimate.total:.2f}")

# Step 3: Execute
if result.is_safe:
    ex = execute(result)
    print(f"\nResult: {ex.rows_extracted:,} rows in {ex.duration_seconds:.1f}s")
    print(f"  Throughput: {ex.rows_extracted / ex.duration_seconds:,.0f} rows/sec")

    # Step 4: Anomaly check vs local baseline
    from ixtract.diagnosis import detect_anomaly
    local_baseline = [856_000, 850_000, 826_000, 824_000, 811_000, 806_000]
    anomaly = detect_anomaly(ex.rows_extracted / ex.duration_seconds, local_baseline)
    print(f"\n  Anomaly: {anomaly.is_anomaly} — {anomaly.message}")