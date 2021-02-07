from functools import reduce
import itertools
import numpy as np
from numpy.random import default_rng
import time
from typing import List, Tuple, Dict

import workflow
from workflow import ExperimentJob
from sched_model_v2 import (
    System,
    Job,
    fcfs,
    easy_backfill,
    conservative_backfill,
    hybrid_backfill,
)

ats_jobs = list(workflow.read_ats_trace("./test-workflow-no40.txt"))
rng = default_rng()


class ModelSleepJob(Job):
    def __init__(self, job: ExperimentJob):
        super().__init__(job.timelimit, np.array([job.cores]))
        self.actual_runtime = 60 if job.cores > 1 else 10
        self.actual_runtime += int(rng.normal(0, 2))

        if self.actual_runtime < 0:
            self.actual_runtime = 0

    def compute_actual_runtime(self, _system: System) -> int:
        return self.actual_runtime


def setup_systems(
    total_cores: int, topology: Tuple[int, ...], dist_method
) -> Dict[str, System]:
    num_leaves = reduce(lambda x, y: x * y, topology, 1)
    leaf_cores, r = divmod(total_cores, num_leaves)
    assert r == 0, "cores not divisible by leaf count"

    leaves: Dict[str, List[ExperimentJob]] = dist_method(topology, ats_jobs)
    system_models = {}

    for tree_id, job_list in leaves.items():
        system = System(np.array([leaf_cores]))
        for job in job_list:
            system.enqueue_job(ModelSleepJob(job))
        system_models[tree_id] = system

    return system_models


def run_model(
    name: str, total_cores: int, topology: Tuple[int, ...], dist_method, policy
) -> Tuple[int, float]:
    systems = setup_systems(total_cores, topology, dist_method)
    makespans = {}

    total_jobs = len(ats_jobs)
    start_time = time.perf_counter()

    prev_finished_count = 0
    for tree_id, system in systems.items():
        while system.tick(policy):
            print(
                "\r"
                + name
                + ": {:<6.1%}".format(
                    (prev_finished_count + len(system.finished_jobs)) / total_jobs
                ),
                end="\r",
            )
        makespans[tree_id] = system.cur_time
        prev_finished_count += len(system.finished_jobs)

    end_time = time.perf_counter()

    return max(makespans.values()), end_time - start_time


def print_test(
    name: str, total_cores: int, topology: Tuple[int, ...], dist_method, policy
):
    makespan, rt = run_model(name, total_cores, topology, dist_method, policy)

    print(
        name
        + ": {makespan:4d}s (calc time: {rt:6.2f})".format(makespan=makespan, rt=rt),
        flush=True,
    )


dist_methods = {
    "Round-Robin": workflow.distribute_rr,
    "Equal Groups": workflow.distribute_by_cores,
    "Prefix-Sum": workflow.distribute_by_utilization,
}

policies = {
    "FCFS": fcfs,
    "EASY": easy_backfill,
    "Hybrid(2)": hybrid_backfill(2),
    "Hybrid(10)": hybrid_backfill(10),
}

if __name__ == "__main__":
    print("64 Cores, Topology 1:")

    for p, d in itertools.product(policies.items(), dist_methods.items()):
        name = "{policy:12s} + {dist:12s}".format(dist=d[0], policy=p[0])
        print_test(name, 64, (1,), d[1], p[1])

    print("\n64 Cores, Topology 1x2:")
    for p, d in itertools.product(policies.items(), dist_methods.items()):
        name = "{policy:12s} + {dist:12s}".format(dist=d[0], policy=p[0])
        print_test(name, 64, (1, 2), d[1], p[1])

    print("\n64 Cores, Topology 1x4:")
    for p, d in itertools.product(policies.items(), dist_methods.items()):
        name = "{policy:12s} + {dist:12s}".format(dist=d[0], policy=p[0])
        print_test(name, 64, (1, 4), d[1], p[1])

    print("\n1280 Cores, Topology 1:")
    for p, d in itertools.product(policies.items(), dist_methods.items()):
        name = "{policy:12s} + {dist:12s}".format(dist=d[0], policy=p[0])
        print_test(name, 1280, (1,), d[1], p[1])

    print("\n1280 Cores, Topology 1x32:")
    for p, d in itertools.product(policies.items(), dist_methods.items()):
        name = "{policy:12s} + {dist:12s}".format(dist=d[0], policy=p[0])
        print_test(name, 1280, (1, 32), d[1], p[1])

    print("\n1280 Cores, Topology 1x32x2:")
    for p, d in itertools.product(policies.items(), dist_methods.items()):
        name = "{policy:12s} + {dist:12s}".format(dist=d[0], policy=p[0])
        print_test(name, 1280, (1, 32, 2), d[1], p[1])

