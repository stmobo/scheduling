from __future__ import annotations

import numpy as np
from typing import Optional

from .resource import Resources, RscCompatible


class Job(object):
    NEW = 0
    PENDING = 1
    STARTED = 2
    RESERVED = 3
    FINISHED = 4

    def __init__(self, timelimit: int, resources: RscCompatible):
        self.timelimit: int = int(timelimit)
        self.resources: Resources = Resources(resources)

        self._job_id: Optional[int] = None
        self.start_time: Optional[int] = None
        self.end_time: Optional[int] = None
        self.deadline: Optional[int] = None
        self._state: int = Job.NEW

        assert self.timelimit > 0

    def compute_actual_runtime(self, system) -> int:
        """This method is called to compute the actual runtime of this job,
        when it is being started.

        This method is intended to be overridden by subclasses.
        """
        return self.timelimit

    @property
    def job_id(self) -> Optional[int]:
        return self._job_id

    @property
    def is_new(self) -> bool:
        return self._state == Job.NEW

    @property
    def is_pending(self) -> bool:
        return self._state == Job.PENDING

    @property
    def is_running(self) -> bool:
        return self._state == Job.STARTED

    @property
    def is_reserved(self) -> bool:
        return self._state == Job.RESERVED

    @property
    def is_finished(self) -> bool:
        return self._state == Job.FINISHED

    def enqueued(self, job_id: int):
        """Move this job into the PENDING state."""
        self._job_id = job_id
        self._state = Job.PENDING

    def reserve(self, start_time: int):
        """Move this job into the RESERVED state."""
        self.start_time = start_time
        self.deadline = start_time + self.timelimit
        self._state = Job.RESERVED

    def unreserve(self):
        """Move this job out of the RESERVED state, and into the PENDING state."""
        self.start_time = None
        self.deadline = None
        self._state = Job.PENDING

    def start(self, system):
        """Move this job into the STARTED state."""
        self.start_time = system.cur_time
        self.deadline = system.cur_time + self.timelimit
        self.end_time = system.cur_time + int(self.compute_actual_runtime(system))
        self._state = Job.STARTED

    def end(self, end_time: int):
        """Move this job into the FINISHED state."""
        self.end_time = end_time
        self._state = Job.FINISHED

    def __str__(self) -> str:
        return "Job({}, state={})".format(self.job_id, self._state)

    def __repr__(self) -> str:
        return self.__str__()
