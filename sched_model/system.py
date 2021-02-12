from __future__ import annotations

from collections import deque
from typing import List, Deque, Optional, Dict, Callable, Iterator, Tuple, Set

from .resource import Resources, RscCompatible
from .job import Job
from .tree import RBTree
from .tree.base import TreeNode


class TimelineData(object):
    def __init__(self, resources: Resources):
        self.start: Set[Job] = set()
        self.end: Set[Job] = set()
        self.expired: Set[Job] = set()
        self.resources: Resources = Resources(resources)


class Timeline(object):
    def __init__(self, base_resources: Resources):
        self._total_resources: Resources = Resources(base_resources)
        self._tree: RBTree[int, TimelineData] = RBTree()

    def _get_data(self, t: int) -> TreeNode[int, TimelineData]:
        insert, tree_node = self._tree.get_or_insert_node(t)
        if insert:
            if tree_node.prev is not None:
                prev_data: TimelineData = tree_node.prev.value
                prev_rsc = prev_data.resources
            else:
                prev_rsc = self._total_resources
            tree_node.value = TimelineData(prev_rsc)
        return tree_node

    def _insert_start_event(self, t: int, job: Job):
        node = self._get_data(t)
        data: TimelineData = node.value
        data.start.add(job)

    def _insert_expire_event(self, t: int, job: Job):
        node = self._get_data(t)
        data: TimelineData = node.value
        data.expired.add(job)

    def _insert_end_event(self, t: int, job: Job):
        node = self._get_data(t)
        data: TimelineData = node.value
        data.end.add(job)

    def _cleanup_node(self, node: TreeNode[int, TimelineData]):
        k = node.key
        data: TimelineData = node.value
        if len(data.end) == 0 and len(data.expired) == 0 and len(data.start) == 0:
            del self._tree[k]

    def _remove_start_event(self, t: int, job: Job):
        node = self._get_data(t)
        data: TimelineData = node.value
        data.start.remove(job)
        self._cleanup_node(node)

    def _remove_expire_event(self, t: int, job: Job):
        node = self._get_data(t)
        data: TimelineData = node.value
        data.expired.remove(job)
        self._cleanup_node(node)

    def _remove_end_event(self, t: int, job: Job):
        node = self._get_data(t)
        data: TimelineData = node.value
        data.end.remove(job)
        self._cleanup_node(node)

    def add_job_reservation(self, job: Job):
        self._insert_start_event(job.start_time, job)
        self._insert_expire_event(job.deadline, job)

        for tl_node in self._tree.values(job.start_time, job.deadline):
            tl_node.resources -= job.resources

    def remove_job_reservation(self, job: Job):
        self._remove_start_event(job.start_time, job)
        self._remove_expire_event(job.deadline, job)

        for tl_node in self._tree.values(job.start_time, job.deadline):
            tl_node.resources += job.resources

    def start_job_reservation(self, job: Job):
        self._insert_end_event(job.end_time, job)

    def end_job_reservation(self, job: Job, new_end_time: int):
        prev_end_time = job.end_time
        prev_deadline = job.deadline

        assert new_end_time <= prev_deadline
        assert new_end_time <= prev_end_time

        if new_end_time < prev_end_time:
            self._insert_end_event(new_end_time, job)
            self._remove_end_event(prev_end_time, job)

        if new_end_time < prev_deadline:
            for node_data in self._tree.values(new_end_time, prev_deadline):
                node_data.resources += job.resources

        self._remove_expire_event(prev_deadline, job)

    def iter_resources(
        self, start_time: int, end_time: Optional[int] = None, copy: bool = True
    ) -> Iterator[Tuple[int, Resources]]:
        if len(self._tree) == 0:
            yield (start_time, self._total_resources.clone())
            return

        iter_start_key = self._tree.upper_bound(start_time + 1)
        if iter_start_key is not None:
            iter_start_key = iter_start_key[0]

        for t, data in self._tree.items(iter_start_key, end_time):
            if copy:
                yield (max(start_time, t), data.resources.clone())
            else:
                yield (max(start_time, t), data.resources)

    def can_schedule(self, job: Job, start_time: int) -> bool:
        if len(self._tree) == 0:
            return True

        return all(
            rsc.all_geq(job.resources)
            for _, rsc in self.iter_resources(start_time, start_time + job.timelimit)
        )

    def find_schedulable_time(
        self, job: Job, start_time: int, reserve: bool
    ) -> Optional[int]:
        if len(self._tree) == 0:
            return start_time

        iter_start_key = self._tree.upper_bound(start_time + 1)
        if iter_start_key is not None:
            iter_start_key = iter_start_key[0]

        for iter_t in self._tree.keys(iter_start_key, None):
            if (not reserve) and (iter_t > start_time):
                return None

            cur_t = max(start_time, iter_t)
            for data in self._tree.values(iter_t, cur_t + job.timelimit):
                if not data.resources.all_geq(job.resources):
                    break
            else:
                return cur_t
        else:
            raise RuntimeError("could not find job scheduling time")

    def iter(self, *args, **kwargs) -> Iterator[Tuple[int, TimelineData]]:
        return self._tree.items(*args, **kwargs)

    def next_event(self, after_time: int) -> Optional[Tuple[int, TimelineData]]:
        return self._tree.lower_bound(after_time + 1)


class System(object):
    def __init__(self, resources: RscCompatible):
        self.total_resources: Resources = Resources(resources)
        self.cur_time: int = 0

        self._jobs_enqueued: int = 0
        self._should_run_sched_loop: bool = False

        self.pending_jobs: Deque[Job] = deque()
        self.finished_jobs: Deque[Job] = deque()
        self.reserved_jobs: List[Job] = []
        self._timeline: Timeline = Timeline(self.total_resources)

    @property
    def should_run_sched_loop(self) -> bool:
        return self._should_run_sched_loop

    def iter_timeline(self, *args, **kwargs) -> Iterator[Tuple[int, TimelineData]]:
        return self._timeline.iter(*args, **kwargs)

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
        if not was_reserved:
            self._timeline.add_job_reservation(job)
        self._timeline.start_job_reservation(job)

        self._should_run_sched_loop = True

    def _end_job(self, job: Job):
        """End a `STARTED` job at the current system timestep.

        Jobs can be ended before their slated end_time or deadline; the
        timeline will be updated as appropriate.
        """

        assert self.cur_time >= job.start_time
        assert job.is_running

        self._timeline.end_job_reservation(job, self.cur_time)
        job.end(self.cur_time)
        self.finished_jobs.append(job)
        self._should_run_sched_loop = True

    def _reserve_job(self, job: Job, t: int):
        """Insert a reservation for a `PENDING` job at the given timestep."""
        assert t > self.cur_time
        assert job.is_pending

        job.reserve(t)
        self._timeline.add_job_reservation(job)
        self.reserved_jobs.append(job)

    def unreserve_all_jobs(self):
        """Clear all job reservations.
        
        The previously-reserved jobs will be added back onto the pending job
        queue in order.
        """
        self.reserved_jobs.sort(key=lambda j: j.job_id, reverse=True)
        for j in self.reserved_jobs:
            assert j.is_reserved

            self._timeline.remove_job_reservation(j)
            j.unreserve()
            self.pending_jobs.appendleft(j)

        self.reserved_jobs = []

    def can_schedule(self, job: Job, start_time: int) -> bool:
        """Check whether a job can be started at a given time."""
        return self._timeline.can_schedule(job, start_time)

    def start_or_reserve_job(self, job: Job, reserve: bool) -> int:
        """Try to start a job, optionally creating a reservation if not possible.
        
        This method returns the new state of the job.
        """

        schedule_tm = self._timeline.find_schedulable_time(job, self.cur_time, reserve)

        if schedule_tm is None:
            return Job.PENDING
        elif schedule_tm > self.cur_time:
            self._reserve_job(job, schedule_tm)
            return Job.RESERVED
        elif schedule_tm == self.cur_time:
            self._start_job(job)
            return Job.STARTED
        else:
            raise RuntimeError("Job was scheduled in the past?")

    def run_sched_loop(self, sched_policy: Callable[[System], None]):
        if self._should_run_sched_loop:
            sched_policy(self)
            self._should_run_sched_loop = False

    def handle_events(self):
        try:
            self.cur_time, node = self._timeline.next_event(self.cur_time)
        except TypeError:
            return False

        # note: these methods may modify the lists in this node, so iterate over
        # copies of the lists in this node instead

        for j in list(node.start):
            assert j.is_reserved
            self._start_job(j)

        for j in list(node.end):
            self._end_job(j)

        for j in list(node.expired):
            self._end_job(j)

        self._should_run_sched_loop = True
        return True

    def tick(self, sched_policy: Callable[[System], None]):
        """Advance to the next timestep, handle job events, and run scheduler
        loop iterations as necessary.
        
        Returns whether there was a next timestep to advance to.
        (If you are calling this in a loop, you can stop once this returns False.)
        """
        self.run_sched_loop(sched_policy)
        if self.handle_events():
            self.run_sched_loop(sched_policy)
            return True

    def run(self, sched_policy: Callable[[System], None]):
        while self.tick(sched_policy):
            pass
