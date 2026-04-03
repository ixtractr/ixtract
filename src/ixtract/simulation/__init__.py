"""Simulation Engine — synthetic data source for controller validation.

Produces realistic extraction metrics without any database. The controller,
deviation analyzer, and state store use the exact same interfaces as real runs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

from ixtract.diagnosis import RunMetrics


@dataclass
class SimulationConfig:
    total_rows: int = 10_000_000
    row_bytes: int = 200
    growth_rate_per_run: float = 0.0
    peak_throughput_rows_sec: float = 50_000.0
    optimal_workers: int = 6
    concurrency_curve: Literal["logarithmic", "linear", "plateau_decline"] = "logarithmic"
    base_query_latency_ms: float = 50.0
    latency_jitter_pct: float = 0.10
    skew_distribution: Literal["uniform", "normal", "power_law"] = "uniform"
    skew_intensity: float = 0.0
    chunk_error_rate: float = 0.0
    latency_spike_on_run: int = -1
    latency_spike_multiplier: float = 1.5
    seed: int = 42


class SimulatedSource:
    """Configurable synthetic extraction source."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._run_num = 0
        self._current_rows = config.total_rows

    def run(
        self, worker_count: int, chunk_count: int = 10,
        previous_throughput: float = 0.0, previous_workers: int = 0,
    ) -> RunMetrics:
        self._run_num += 1
        cfg = self.config

        # Growth
        if self._run_num > 1 and cfg.growth_rate_per_run > 0:
            self._current_rows = int(self._current_rows * (1 + cfg.growth_rate_per_run))

        # Throughput at this worker count
        raw_tp = self._throughput_at(worker_count)

        # Latency spike
        if cfg.latency_spike_on_run == self._run_num:
            raw_tp /= cfg.latency_spike_multiplier

        # Jitter
        jitter = 1.0 + self._rng.uniform(-cfg.latency_jitter_pct, cfg.latency_jitter_pct)
        eff_tp = raw_tp * jitter

        total_bytes = self._current_rows * cfg.row_bytes
        duration = self._current_rows / max(eff_tp, 1.0)

        chunk_durs = self._chunk_durations(chunk_count, duration)
        idle_pcts = self._worker_idle(worker_count, chunk_count)

        query_ms = cfg.base_query_latency_ms
        if cfg.latency_spike_on_run == self._run_num:
            query_ms *= cfg.latency_spike_multiplier

        return RunMetrics(
            total_rows=self._current_rows,
            total_bytes=total_bytes,
            duration_seconds=duration,
            worker_count=worker_count,
            avg_throughput_rows_sec=eff_tp,
            chunk_durations=tuple(chunk_durs),
            worker_idle_pcts=tuple(idle_pcts),
            source_query_ms_avg=query_ms,
            previous_throughput_rows_sec=previous_throughput,
            previous_worker_count=previous_workers,
            worker_count_changed=(worker_count != previous_workers),
        )

    def _throughput_at(self, workers: int) -> float:
        cfg = self.config
        opt, peak = cfg.optimal_workers, cfg.peak_throughput_rows_sec

        if cfg.concurrency_curve == "linear":
            if workers <= opt:
                return peak * (workers / opt)
            return peak * max(0.2, 1.0 - 0.08 * (workers - opt))

        if cfg.concurrency_curve == "logarithmic":
            if workers <= opt:
                return peak * (math.log(1 + workers) / math.log(1 + opt))
            return peak * max(0.2, 1.0 - 0.10 * (workers - opt))

        if cfg.concurrency_curve == "plateau_decline":
            if workers <= opt:
                return peak * min(1.0, 0.3 + 0.7 * (workers / opt))
            return peak * max(0.2, 1.0 - 0.15 * (workers - opt))

        return peak

    def _chunk_durations(self, n: int, total: float) -> list[float]:
        cfg, rng = self.config, self._rng
        base = total / max(n, 1)

        if cfg.skew_distribution == "uniform" or cfg.skew_intensity <= 0:
            return [base * rng.uniform(0.9, 1.1) for _ in range(n)]
        if cfg.skew_distribution == "normal":
            sigma = base * cfg.skew_intensity * 0.5
            return [max(base * 0.1, rng.gauss(base, sigma)) for _ in range(n)]
        if cfg.skew_distribution == "power_law":
            out = []
            for _ in range(n):
                if rng.random() < cfg.skew_intensity * 0.2:
                    out.append(base * rng.uniform(3.0, 8.0))
                else:
                    out.append(base * rng.uniform(0.5, 1.2))
            return out
        return [base] * n

    def _worker_idle(self, workers: int, chunks: int) -> list[float]:
        rng = self._rng
        if chunks <= workers:
            return [
                rng.uniform(0.0, 0.1) if i < chunks else rng.uniform(0.7, 0.95)
                for i in range(workers)
            ]
        return [rng.uniform(0.02, 0.15) for _ in range(workers)]

    def reset(self) -> None:
        self._rng = random.Random(self.config.seed)
        self._run_num = 0
        self._current_rows = self.config.total_rows
