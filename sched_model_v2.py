from __future__ import annotations

import bisect
from collections import deque
from functools import reduce, partial
import numpy as np
from typing import List, Deque, Optional, Union, Dict, Callable, Iterator, Tuple

from avl import AVLTree


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
    PENDING = 0
    STARTED = 1
    RESERVED = 2

    def __init__(self, t: int, resources: RscCompatible):
        self.runtime: int = int(t)
        self.resources: Resources = Resources(resources)

        self.job_id: Optional[int] = None
        self.start_time: Optional[int] = None
        self.reserved: bool = False

    @property
    def end_time(self) -> Optional[int]:
        if self.start_time is None:
            return None
        else:
            return self.start_time + self.runtime

    def __str__(self) -> str:
        return "Job({})".format(self.job_id)

    def __repr__(self) -> str:
        return "Job({})".format(self.job_id)


class System(object):
    def __init__(self, resources: RscCompatible):
        self.total_resources: Resources = Resources(resources)
        self.cur_time: int = 0

        self._cur_resources: Resources = Resources(resources)
        self._jobs_enqueued: int = 0

        self.pending_jobs: Deque[Job] = deque()
        self.finished_jobs: Deque[Job] = deque()
        self.reserved_jobs: List[Job] = []
        self._timeline: AVLTree[int, Dict[str, List[Job]]] = AVLTree()

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

    def enqueue_job(self, job: Job):
        if not self.total_resources.all_geq(job.resources):
            raise ValueError("Job resource requirements cannot be satisfied")

        job.job_id = self._jobs_enqueued
        self._jobs_enqueued += 1
        self.pending_jobs.append(job)

    def iter_timeline(
        self, *args, **kwargs
    ) -> Iterator[Tuple[int, Dict[str, List[Job]]]]:
        return self._timeline.items(*args, **kwargs)

    def available_resources_at(self, t: Optional[int] = None) -> Resources:
        rsc = self._cur_resources.clone()
        if t is None:
            return rsc

        for node in self._timeline.values(self.cur_time + 1, t + 1):
            for job in node.get("start", []):
                rsc -= job.resources

            for job in node.get("end", []):
                rsc += job.resources

        return rsc

    def can_schedule(self, job: Job) -> bool:
        rsc = self._cur_resources - job.resources
        if not rsc.valid():
            return False

        # Check resource availability throughout job runtime:
        end_time = self.cur_time + job.runtime
        for node in self._timeline.values(self.cur_time + 1, end_time):
            for job in node.get("start", []):
                rsc -= job.resources

            for job in node.get("end", []):
                rsc += job.resources

            if not rsc.valid():
                return False

        return True

    def start_job(self, job: Job, reserve: bool) -> int:
        if self.can_schedule(job):
            job.start_time = self.cur_time
            self._cur_resources -= job.resources
            self._insert_timeline_event(self.cur_time, "start", job)
            self._insert_timeline_event(job.end_time, "end", job)

            return Job.STARTED
        elif reserve:
            rsvp_rsc = self._cur_resources.clone()
            rsvp_time = -1

            for t, node in self._timeline.items(self.cur_time + 1):
                for j in node.get("start", []):
                    rsvp_rsc -= j.resources

                for j in node.get("end", []):
                    rsvp_rsc += j.resources

                # Check resource availability throughout job runtime:
                tmp_rsc = rsvp_rsc - job.resources
                if not tmp_rsc.valid():
                    continue

                for n2 in self._timeline.values(t + 1, t + job.runtime):
                    for j in n2.get("start", []):
                        tmp_rsc -= j.resources

                    for j in n2.get("end", []):
                        tmp_rsc += j.resources

                    if not tmp_rsc.valid():
                        break
                else:
                    # Reservation is valid.
                    rsvp_time = t
                    break
            else:
                raise RuntimeError("Could not find reserve time for job")

            job.start_time = rsvp_time
            job.reserved = True
            self._insert_timeline_event(rsvp_time, "start", job)
            self._insert_timeline_event(job.end_time, "end", job)
            self.reserved_jobs.append(job)

            return Job.RESERVED

        return Job.PENDING

    def unreserve_all_jobs(self):
        self.reserved_jobs.sort(key=lambda j: j.job_id, reverse=True)
        for j in self.reserved_jobs:
            self._remove_timeline_event(j.start_time, "start", j)
            self._remove_timeline_event(j.end_time, "end", j)
            j.start_time = None
            j.reserved = False
            self.pending_jobs.appendleft(j)
        self.reserved_jobs = []

    def tick(self):
        try:
            self.cur_time, node = next(self._timeline.items(self.cur_time + 1))
        except StopIteration:
            return False

        for j in node.get("end", []):
            self.finished_jobs.append(j)
            self._cur_resources += j.resources

        for j in node.get("start", []):
            self._cur_resources -= j.resources
            if j.reserved:
                self.reserved_jobs.remove(j)
                j.reserved = False

        return True

    def run(self, sched_policy: Callable[[System], None]):
        sched_policy(self)
        while self.tick():
            sched_policy(self)


def fcfs(system: System):
    while len(system.pending_jobs) > 0:
        j = system.pending_jobs[0]

        if system.start_job(j, False) != Job.STARTED:
            break

        system.pending_jobs.popleft()


def _backfill_sched(max_backfill: Optional[int], system: System):
    cur_reserved = 0
    new_pending = deque()

    system.unreserve_all_jobs()

    while len(system.pending_jobs) > 0:
        j = system.pending_jobs.popleft()

        if max_backfill is None or (cur_reserved < max_backfill):
            status = system.start_job(j, True)

            assert status != Job.PENDING
            if status == Job.RESERVED:
                cur_reserved += 1
        else:
            status = system.start_job(j, False)

            assert status != Job.RESERVED
            if status == Job.PENDING:
                new_pending.append(j)

    system.pending_jobs = new_pending


easy_backfill = partial(_backfill_sched, 1)
conservative_backfill = partial(_backfill_sched, None)


def hybrid_backfill(max_backfill: int) -> Callable[[System], None]:
    return partial(_backfill_sched, max_backfill)


if __name__ == "__main__":
    system = System(np.array([5]))

    system.enqueue_job(Job(10, np.array([2])))
    system.enqueue_job(Job(5, np.array([3])))
    system.enqueue_job(Job(5, np.array([5])))
    system.enqueue_job(Job(3, np.array([3])))
    system.enqueue_job(Job(3, np.array([1])))
    system.enqueue_job(Job(2, np.array([2])))
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
