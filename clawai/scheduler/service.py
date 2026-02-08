"""
Runtime scheduler service.

This module implements the **time-based event scheduler** for ClawAI Agent OS.

Design principles:
    - Runtime lifecycle integration
    - Event-driven scheduling
    - Persistent job store
    - Deterministic execution
    - Fault isolation
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional, Sequence

from loguru import logger

from clawai.scheduler.types import (
    CronJob,
    CronJobState,
    CronPayload,
    CronSchedule,
    CronStore,
)

# ============================================================
# Constants
# ============================================================

DEFAULT_TICK_INTERVAL_S = 1.0
DEFAULT_STORE_FLUSH_INTERVAL_S = 5.0


# ============================================================
# Types
# ============================================================

CronCallback = Callable[[CronJob], Awaitable[Optional[str]]]


# ============================================================
# Utilities
# ============================================================

def now_ms() -> int:
    return int(time.time() * 1000)


def compute_next_run(schedule: CronSchedule, base_ms: int) -> Optional[int]:
    """
    Compute next run timestamp.

    Supports:
        - at
        - every
        - cron
    """
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > base_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        return base_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from croniter import croniter
            return int(croniter(schedule.expr, base_ms / 1000).get_next() * 1000)
        except Exception:
            logger.exception("Invalid cron expression: {}", schedule.expr)

    return None


# ============================================================
# Runtime Scheduler Kernel
# ============================================================

class CronService:
    """
    Runtime cron scheduler.

    This is a **core runtime service**, not a utility.

    Responsibilities:
        - Persistent scheduling
        - Event-driven wakeup
        - Task execution orchestration
        - Agent runtime triggering
    """

    def __init__(
        self,
        store_path: Path,
        callback: CronCallback,
    ):
        self.store_path = store_path
        self.callback = callback

        self._store: Optional[CronStore] = None
        self._running = False

        self._scheduler_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None

        self._wakeup_event = asyncio.Event()

    # ============================================================
    # Lifecycle
    # ============================================================

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._load_store()
        self._recompute_all()

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._flush_task = asyncio.create_task(self._flush_loop())

        logger.info(
            "Cron scheduler started | jobs={} store={}",
            len(self._store.jobs),
            self.store_path,
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        for task in (self._scheduler_task, self._flush_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._save_store()
        logger.info("Cron scheduler stopped")

    # ============================================================
    # Scheduler Loop
    # ============================================================

    async def _scheduler_loop(self) -> None:
        try:
            while self._running:
                now = now_ms()

                jobs = self._due_jobs(now)
                if jobs:
                    await self._run_jobs(jobs)

                timeout = self._compute_sleep_interval(now)
                try:
                    await asyncio.wait_for(self._wakeup_event.wait(), timeout)
                except asyncio.TimeoutError:
                    pass
                finally:
                    self._wakeup_event.clear()

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Cron scheduler crashed")

    async def _flush_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(DEFAULT_STORE_FLUSH_INTERVAL_S)
                self._save_store()
        except asyncio.CancelledError:
            pass

    # ============================================================
    # Execution
    # ============================================================

    async def _run_jobs(self, jobs: Sequence[CronJob]) -> None:
        for job in jobs:
            await self._execute_job(job)

        self._recompute_all()
        self._wakeup_event.set()

    async def _execute_job(self, job: CronJob) -> None:
        logger.info("Cron executing | {} ({})", job.name, job.id)
        start = now_ms()

        try:
            result = await self.callback(job)

            job.state.last_status = "ok"
            job.state.last_error = None
            logger.success("Cron finished | {} ({})", job.name, job.id)

        except Exception as e:
            job.state.last_status = "error"
            job.state.last_error = str(e)
            logger.exception("Cron failed | {} ({})", job.name, job.id)

        job.state.last_run_at_ms = start
        job.updated_at_ms = now_ms()

        if job.schedule.kind == "at":
            if job.delete_after_run:
                self._store.jobs = [j for j in self._store.jobs if j.id != job.id]
            else:
                job.enabled = False

    # ============================================================
    # Scheduling Logic
    # ============================================================

    def _due_jobs(self, now: int) -> list[CronJob]:
        return [
            j for j in self._store.jobs
            if j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms
        ]

    def _recompute_all(self) -> None:
        now = now_ms()
        for job in self._store.jobs:
            if job.enabled:
                job.state.next_run_at_ms = compute_next_run(job.schedule, now)

    def _compute_sleep_interval(self, now: int) -> float:
        times = [
            j.state.next_run_at_ms
            for j in self._store.jobs
            if j.enabled and j.state.next_run_at_ms
        ]

        if not times:
            return DEFAULT_TICK_INTERVAL_S

        delay = min(times) - now
        return max(delay / 1000, DEFAULT_TICK_INTERVAL_S)

    # ============================================================
    # Store
    # ============================================================

    def _load_store(self) -> None:
        if self.store_path.exists():
            try:
                self._store = CronStore.load(self.store_path)
                return
            except Exception:
                logger.exception("Failed to load cron store")

        self._store = CronStore()

    def _save_store(self) -> None:
        if not self._store:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store.dump(self.store_path)

    # ============================================================
    # Public API
    # ============================================================

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        jobs = self._store.jobs
        if not include_disabled:
            jobs = [j for j in jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float("inf"))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        payload: CronPayload,
        delete_after_run: bool = False,
    ) -> CronJob:
        now = now_ms()

        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=payload,
            state=CronJobState(
                next_run_at_ms=compute_next_run(schedule, now)
            ),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
        )

        self._store.jobs.append(job)
        self._save_store()
        self._wakeup_event.set()

        logger.info("Cron added | {} ({})", name, job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        before = len(self._store.jobs)
        self._store.jobs = [j for j in self._store.jobs if j.id != job_id]
        removed = len(self._store.jobs) < before

        if removed:
            self._save_store()
            self._wakeup_event.set()
            logger.info("Cron removed | {}", job_id)

        return removed

    async def run_job(self, job_id: str) -> bool:
        for job in self._store.jobs:
            if job.id == job_id:
                await self._execute_job(job)
                self._save_store()
                self._wakeup_event.set()
                return True
        return False

    def status(self) -> dict:
        return {
            "running": self._running,
            "jobs": len(self._store.jobs),
            "next_run_at_ms": min(
                (j.state.next_run_at_ms for j in self._store.jobs if j.state.next_run_at_ms),
                default=None,
            ),
        }
