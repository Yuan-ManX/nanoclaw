"""
Cron scheduling domain model.

This module defines the **runtime scheduling primitives** used by
ClawAI Agent OS.

Design goals:
    - Strong semantic modeling
    - Runtime-first abstraction
    - Deterministic scheduling behavior
    - Extensible execution payload
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Any, Dict, List


# ============================================================
# Schedule Model
# ============================================================

ScheduleKind = Literal["at", "every", "cron"]


@dataclass(slots=True)
class CronSchedule:
    """
    Schedule definition.

    Supports:
        - at     → run once at fixed timestamp
        - every  → fixed interval execution
        - cron   → cron expression
    """

    kind: ScheduleKind

    # For "at"
    at_ms: Optional[int] = None

    # For "every"
    every_ms: Optional[int] = None

    # For "cron"
    expr: Optional[str] = None
    tz: Optional[str] = None

    # ----------------------------
    # Validation
    # ----------------------------

    def validate(self) -> None:
        if self.kind == "at":
            if not self.at_ms:
                raise ValueError("schedule.kind='at' requires at_ms")

        elif self.kind == "every":
            if not self.every_ms or self.every_ms <= 0:
                raise ValueError("schedule.kind='every' requires every_ms > 0")

        elif self.kind == "cron":
            if not self.expr:
                raise ValueError("schedule.kind='cron' requires expr")

        else:
            raise ValueError(f"Unknown schedule kind: {self.kind}")


# ============================================================
# Payload Model
# ============================================================

PayloadKind = Literal[
    "agent_turn",
    "system_event",
    "flow_trigger",
]


@dataclass(slots=True)
class CronPayload:
    """
    Execution payload.

    Defines **what happens** when a cron job fires.

    Supported kinds:
        - agent_turn    → normal agent reasoning step
        - system_event  → runtime system event
        - flow_trigger  → start a Flow DAG
    """

    kind: PayloadKind = "agent_turn"

    # Main content
    message: str = ""

    # Delivery routing
    deliver: bool = False
    channel: Optional[str] = None
    to: Optional[str] = None

    # Future extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Runtime State
# ============================================================

JobStatus = Literal["pending", "running", "ok", "error", "skipped"]


@dataclass(slots=True)
class CronJobState:
    """
    Runtime state of scheduled job.

    This structure represents the **runtime lifecycle** of a job.
    """

    next_run_at_ms: Optional[int] = None
    last_run_at_ms: Optional[int] = None
    last_status: Optional[JobStatus] = None
    last_error: Optional[str] = None

    # Execution metrics
    run_count: int = 0

    # ----------------------------
    # State transitions
    # ----------------------------

    def mark_running(self) -> None:
        self.last_status = "running"

    def mark_success(self) -> None:
        self.last_status = "ok"
        self.last_error = None
        self.run_count += 1

    def mark_error(self, err: Exception | str) -> None:
        self.last_status = "error"
        self.last_error = str(err)
        self.run_count += 1

    def mark_skipped(self) -> None:
        self.last_status = "skipped"


# ============================================================
# Job Model
# ============================================================

@dataclass(slots=True)
class CronJob:
    """
    Cron job runtime object.

    This is the **atomic scheduling primitive** of Agent OS.
    """

    id: str
    name: str

    enabled: bool = True

    schedule: CronSchedule = field(
        default_factory=lambda: CronSchedule(kind="every", every_ms=60_000)
    )

    payload: CronPayload = field(default_factory=CronPayload)

    state: CronJobState = field(default_factory=CronJobState)

    created_at_ms: int = 0
    updated_at_ms: int = 0

    # Lifecycle policy
    delete_after_run: bool = False

    # ----------------------------
    # Semantic helpers
    # ----------------------------

    @property
    def is_due(self) -> bool:
        return (
            self.enabled
            and self.state.next_run_at_ms is not None
        )

    def disable(self) -> None:
        self.enabled = False
        self.state.next_run_at_ms = None


# ============================================================
# Store Model
# ============================================================

@dataclass
class CronStore:
    """
    Persistent cron job store.

    This object is intentionally:
        - Serializable
        - Deterministic
        - Append-only friendly
    """

    version: int = 1
    jobs: List[CronJob] = field(default_factory=list)

    # ----------------------------
    # Query helpers
    # ----------------------------

    def get(self, job_id: str) -> Optional[CronJob]:
        for j in self.jobs:
            if j.id == job_id:
                return j
        return None

    def remove(self, job_id: str) -> bool:
        before = len(self.jobs)
        self.jobs = [j for j in self.jobs if j.id != job_id]
        return len(self.jobs) < before

    def enabled_jobs(self) -> List[CronJob]:
        return [j for j in self.jobs if j.enabled]

    # ----------------------------
    # Persistence hooks (optional)
    # ----------------------------

    @classmethod
    def load(cls, path):
        import json
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def dump(self, path):
        import json
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CronStore":
        jobs = []
        for j in data.get("jobs", []):
            jobs.append(
                CronJob(
                    id=j["id"],
                    name=j["name"],
                    enabled=j.get("enabled", True),
                    schedule=CronSchedule(**j["schedule"]),
                    payload=CronPayload(**j["payload"]),
                    state=CronJobState(**j.get("state", {})),
                    created_at_ms=j.get("created_at_ms", 0),
                    updated_at_ms=j.get("updated_at_ms", 0),
                    delete_after_run=j.get("delete_after_run", False),
                )
            )
        return cls(
            version=data.get("version", 1),
            jobs=jobs,
        )
  
