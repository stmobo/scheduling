from __future__ import annotations

from collections import deque
from functools import partial
from typing import Callable, Optional

from .job import Job
from .system import System


def fcfs(system: System):
    while len(system.pending_jobs) > 0:
        j = system.pending_jobs[0]

        if system.start_or_reserve_job(j, False) != Job.STARTED:
            break

        system.pending_jobs.popleft()


def _backfill_sched(max_backfill: Optional[int], system: System):
    cur_reserved = 0
    new_pending = deque()

    system.unreserve_all_jobs()

    while len(system.pending_jobs) > 0:
        j = system.pending_jobs.popleft()

        if max_backfill is None or (cur_reserved < max_backfill):
            status = system.start_or_reserve_job(j, True)

            assert status != Job.PENDING
            if status == Job.RESERVED:
                cur_reserved += 1
        else:
            status = system.start_or_reserve_job(j, False)

            assert status != Job.RESERVED
            if status == Job.PENDING:
                new_pending.append(j)

    system.pending_jobs = new_pending


easy_backfill = partial(_backfill_sched, 1)
conservative_backfill = partial(_backfill_sched, None)


def hybrid_backfill(max_backfill: int) -> Callable[[System], None]:
    return partial(_backfill_sched, max_backfill)