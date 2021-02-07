from hypothesis import given, note, assume, strategies as st

from sched_model_v2 import (
    System,
    Job,
    fcfs,
    easy_backfill,
    conservative_backfill,
    hybrid_backfill,
)
import numpy as np

job_val = st.integers(min_value=1, max_value=np.iinfo(np.int).max)
job_strategy = st.lists(st.tuples(job_val, job_val))


def setup_system(jobs):
    system_resources = max((j[1] for j in jobs), default=1)
    system = System(np.array([system_resources]))

    for tm, resources in jobs:
        system.enqueue_job(Job(tm, np.array([resources])))

    return system


def check_system(system: System):
    assert (
        system._cur_resources.valid()
    ), "Invalid resource count at time {}: {}".format(
        system.cur_time, str(system._cur_resources)
    )

    assert system.total_resources.all_geq(
        system._cur_resources
    ), "Current resource count at time {} is too high: {} > {}".format(
        system.cur_time, str(system._cur_resources), str(system.total_resources)
    )


def run_system(system: System, policy):
    policy(system)
    check_system(system)

    while system.tick():
        check_system(system)
        policy(system)
        check_system(system)


@given(job_strategy)
def test_fcfs(jobs):
    run_system(setup_system(jobs), fcfs)


@given(job_strategy)
def test_easy(jobs):
    run_system(setup_system(jobs), easy_backfill)


@given(job_strategy)
def test_conservative(jobs):
    run_system(setup_system(jobs), conservative_backfill)


@given(job_strategy, st.integers(min_value=2))
def test_hybrid(jobs, max_backfill):
    run_system(setup_system(jobs), hybrid_backfill(max_backfill))

