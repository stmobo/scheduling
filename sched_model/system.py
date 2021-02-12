from __future__ import annotations

from collections import deque
from typing import List, Deque, Optional, Dict, Callable, Iterator, Tuple

from .resource import Resources, RscCompatible
from .job import Job
from .tree import RBTree


class System(object):
    def __init__(self, resources: RscCompatible):
        self.total_resources: Resources = Resources(resources)
        self.cur_time: int = 0

        self._cur_resources: Resources = Resources(resources)
        self._jobs_enqueued: int = 0
        self._should_run_sched_loop: bool = False

        self.pending_jobs: Deque[Job] = deque()
        self.finished_jobs: Deque[Job] = deque()
        self.reserved_jobs: List[Job] = []
        self._timeline: RBTree[int, Dict[str, List[Job]]] = RBTree()

    def _get_timeline_node(self, t: int) -> Dict[str, List[Job]]:
        try:
            return self._timeline[t]
        except KeyError:
            ret = {}
            self._timeline[t] = ret
            return ret

    def _insert_timeline_event(self, t: int, tag: str, job: Job):
        tm = self._get_timeline_node(t)
        if tag in tm:
            tm[tag].append(job)
        else:
            tm[tag] = [job]

    def _remove_timeline_event(self, t: int, tag: str, job: Job):
        tm = self._get_timeline_node(t)
        tm[tag].remove(job)

        if len(tm[tag]) == 0:
            del tm[tag]

        if len(tm) == 0:
            del self._timeline[t]

    def iter_timeline(
        self, *args, **kwargs
    ) -> Iterator[Tuple[int, Dict[str, List[Job]]]]:
        return self._timeline.items(*args, **kwargs)

    def enqueue_job(self, job: Job):
        """Push a `NEW` job onto the pending job queue."""
        assert job.is_new
        if not self.total_resources.all_geq(job.resources):
            raise ValueError("Job resource requirements cannot be satisfied")

        job.enqueued(self._jobs_enqueued)
        self._jobs_enqueued += 1
        self.pending_jobs.append(job)
        self._should_run_sched_loop = True

    def _start_job(self, job: Job):
        """Start a `PENDING` or `RESERVED` job at the current system timestep.
        
        Note that if the given job is in the `RESERVED` state, its reservation
        must be slated to start at the current system timestep.
        """

        assert job.is_reserved or job.is_pending
        was_reserved = job.is_reserved

        if was_reserved:
            # if we're starting this job from reserved state, then "start" and
            # "expiration" events should already be in the timeline
            assert job.start_time == self.cur_time
            self.reserved_jobs.remove(job)

        job.start(self)
        self._cur_resources -= job.resources
        self._insert_timeline_event(job.end_time, "end", job)

        if not was_reserved:
            self._insert_timeline_event(self.cur_time, "start", job)
            self._insert_timeline_event(job.deadline, "expiration", job)
        self._should_run_sched_loop = True

    def _end_job(self, job: Job):
        """End a `STARTED` job at the current system timestep.

        Jobs can be ended before their slated end_time or deadline; the
        timeline will be updated as appropriate.
        """

        assert self.cur_time >= job.start_time
        assert job.is_running

        if self.cur_time != job.end_time:
            self._insert_timeline_event(self.cur_time, "end", job)
            self._remove_timeline_event(job.end_time, "end", job)

        self._cur_resources += job.resources
        self._remove_timeline_event(job.deadline, "expiration", job)
        job.end(self.cur_time)
        self.finished_jobs.append(job)
        self._should_run_sched_loop = True

    def _reserve_job(self, job: Job, t: int):
        """Insert a reservation for a `PENDING` job at the given timestep."""
        assert t > self.cur_time
        assert job.is_pending

        job.reserve(t)
        self._insert_timeline_event(t, "start", job)
        self._insert_timeline_event(job.deadline, "expiration", job)
        self.reserved_jobs.append(job)

    def unreserve_all_jobs(self):
        """Clear all job reservations.
        
        The previously-reserved jobs will be added back onto the pending job
        queue in order.
        """
        self.reserved_jobs.sort(key=lambda j: j.job_id, reverse=True)
        for j in self.reserved_jobs:
            assert j.is_reserved

            self._remove_timeline_event(j.start_time, "start", j)
            self._remove_timeline_event(j.deadline, "expiration", j)
            j.unreserve()
            self.pending_jobs.appendleft(j)

        self.reserved_jobs = []

    def _iter_timeline_resources(
        self, start_rsc: Resources, start_time: int, end_time: Optional[int] = None
    ) -> Iterator[Tuple[int, Resources]]:
        """Walk through the timeline, yielding projected available resources.
        
        This method yields _projected_ resources; it deliberately does not use
        "end" events (which are based on actual runtimes) and only looks at
        "expiration" events (which indicate deadlines).
        """
        rsc = start_rsc.clone()

        for t, node in self._timeline.items(start_time + 1, end_time):
            for j in node.get("start", []):
                rsc -= j.resources

            for j in node.get("expiration", []):
                rsc += j.resources

            yield (t, rsc.clone())

    def can_schedule(self, job: Job, start_rsc: Resources, start_time: int) -> bool:
        """Check whether a job can be started at a given time."""
        rsc = start_rsc - job.resources
        if not rsc.valid():
            return False

        # Check resource availability throughout job runtime:
        end_time = start_time + job.timelimit
        for _, rsc in self._iter_timeline_resources(rsc, start_time, end_time):
            if not rsc.valid():
                return False
        return True

    def start_or_reserve_job(self, job: Job, reserve: bool) -> int:
        """Try to start a job, optionally creating a reservation if not possible.
        
        This method returns the new state of the job.
        """
        if self.can_schedule(job, self._cur_resources, self.cur_time):
            self._start_job(job)
            return Job.STARTED
        elif reserve:
            rsvp_time = -1
            for t, rsc in self._iter_timeline_resources(
                self._cur_resources, self.cur_time
            ):
                if self.can_schedule(job, rsc, t):
                    # Would calling _reserve_job directly here cause problems?
                    # _reserve_job modifies the timeline, and we're iterating
                    # over it, but on the other hand we also would just return
                    # immediately. To be safe, we break here.
                    rsvp_time = t
                    break
            else:
                raise RuntimeError("Could not find time for job reservation")

            self._reserve_job(job, rsvp_time)
            return Job.RESERVED
        else:
            return Job.PENDING

    def tick(self, sched_policy: Callable[[System], None]):
        """Advance to the next timestep, handle job events, and run scheduler
        loop iterations as necessary.
        
        Returns whether there was a next timestep to advance to.
        (If you are calling this in a loop, you can stop once this returns False.)
        """
        if self._should_run_sched_loop:
            sched_policy(self)
            self._should_run_sched_loop = False

        try:
            self.cur_time, node = next(self._timeline.items(self.cur_time + 1))
        except StopIteration:
            return False

        # note: these methods may modify the lists in this node, so iterate over
        # copies of the lists in this node instead

        for j in list(node.get("start", [])):
            assert j.is_reserved
            self._start_job(j)

        for j in list(node.get("end", [])):
            self._end_job(j)

        for j in list(node.get("expiration", [])):
            self._end_job(j)

        sched_policy(self)
        self._should_run_sched_loop = False

        return True

    def run(self, sched_policy: Callable[[System], None]):
        while self.tick(sched_policy):
            pass
