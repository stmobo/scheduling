from __future__ import annotations

import bisect
from collections import deque
from functools import reduce, partial
import numpy as np
from typing import List, Deque, Optional, Union, Dict, Callable, Iterator, Tuple, Set

from tree import RBTree


class Resources(object):
    def __init__(self, v: RscCompatible):
        self.resources: np.ndarray = self._resource_vec(v).astype(np.int)

    def clone(self) -> Resources:
        return Resources(self.resources)

    def valid(self) -> bool:
        return (self.resources >= 0).all()

    @staticmethod
    def _resource_vec(val: RscCompatible) -> np.ndarray:
        """Gets the `resources` vector out of a Resources object, and leaves
        other parameters untouched.
        """
        try:
            return val.resources
        except AttributeError:
            return val

    def all_geq(self, other: RscCompatible) -> bool:
        """Test whether all resources in this vector are greater than or 
        equal to all resources contained in `other`.
        """
        return (self.resources >= self._resource_vec(other)).all()

    def __add__(self, other: RscCompatible) -> Resources:
        return Resources(self.resources + self._resource_vec(other))

    def __sub__(self, other: RscCompatible) -> Resources:
        return Resources(self.resources - self._resource_vec(other))

    def __radd__(self, other: RscCompatible) -> Resources:
        return self.__add__(other)

    def __rsub__(self, other: RscCompatible) -> Resources:
        return Resources(self._resource_vec(other) - self.resources)

    def __iadd__(self, other: RscCompatible) -> Resources:
        self.resources += self._resource_vec(other)
        return self

    def __isub__(self, other: RscCompatible) -> Resources:
        self.resources -= self._resource_vec(other)
        return self

    def __iter__(self):
        return self.resources.__iter__()

    def __len__(self) -> int:
        return len(self.resources)

    def __str__(self) -> str:
        return str(self.resources)

    def __repr__(self) -> str:
        return "Resources(" + repr(self.resources) + ")"


RscCompatible = Union[Resources, np.ndarray]


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

    def compute_actual_runtime(self, system: System) -> int:
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

    def start(self, system: System):
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


if __name__ == "__main__":
    system = System(np.array([6]))

    system.enqueue_job(Job(2, np.array([1])))
    system.enqueue_job(Job(3, np.array([1])))
    system.enqueue_job(Job(5, np.array([2])))
    system.enqueue_job(Job(2, np.array([6])))
    system.enqueue_job(Job(1, np.array([1])))
    system.enqueue_job(Job(1, np.array([1])))
    system.enqueue_job(Job(1, np.array([1])))
    system.run(easy_backfill)

    util_vecs = []
    cur_jobs = []

    for t, node in system.iter_timeline():
        s = ""

        if "start" in node:
            s = "started job{} {}".format(
                "s" if len(node["start"]) > 1 else "",
                ", ".join(map(lambda j: str(j.job_id), node["start"])),
            )

            for j in node["start"]:
                cur_jobs.append(j)

        if "end" in node:
            if "start" in node:
                s += " / "

            s += "finished job{} {}".format(
                "s" if len(node["end"]) > 1 else "",
                ", ".join(map(lambda j: str(j.job_id), node["end"])),
            )

            for j in node["end"]:
                cur_jobs.remove(j)

        util_vecs.append((t, list(cur_jobs), s))

    it = iter(util_vecs)
    prev_tm, prev_jobs, prev_msg = next(it)
    glyph_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    header_1 = "Time |"
    header_2 = "-----|"
    for i, width in enumerate(system.total_resources):
        header_1 += format("R" + str(i), "<" + str(width) + "s") + "|"
        header_2 += ("-" * width) + "|"
    print(header_1 + "\n" + header_2)

    for tm, job_list, msg in it:
        util_bars = [""] * len(system.total_resources)

        for job in prev_jobs:
            glyph = glyph_list[job.job_id % len(glyph_list)]
            for i, util in enumerate(job.resources):
                util_bars[i] += glyph * util

        util_bar = ""
        for sub_bar, max_util in zip(util_bars, system.total_resources):
            util_bar += sub_bar + ("." * (max_util - len(sub_bar))) + "|"

        for t in range(prev_tm, tm):
            if t == prev_tm:
                end = " (%s)\n" % prev_msg
            else:
                end = "\n"

            print("{:<4d} |".format(t) + util_bar, end=end)

        prev_tm = tm
        prev_jobs = job_list
        prev_msg = msg
