from __future__ import annotations

from io import TextIOBase
from itertools import cycle, product, groupby
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Iterator, Iterable

import numpy as np
from numpy import random


class ExperimentJob(object):
    def __init__(
        self, workflow_job_idx: int, args: List[str], cores: int, timelimit: int
    ):
        self.args: List[str] = args
        self.timelimit: int = timelimit
        self.cores: int = cores
        self.workflow_job_idx: int = workflow_job_idx

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentJob:
        return cls(
            data["workflow_idx"], list(data["args"]), data["cores"], data["timelimit"]
        )

    def as_dict(self) -> dict:
        return {
            "args": self.args,
            "timelimit": self.timelimit,
            "cores": self.cores,
            "workflow_idx": self.workflow_job_idx,
        }


def shuffle_iterable(it, shuffle_seed: int, n=None):
    rng = random.default_rng(shuffle_seed)
    items = list(it)

    if n is None:
        while True:
            rng.shuffle(items)
            yield from items
    else:
        for _ in range(n):
            rng.shuffle(items)
            yield from items


def get_leaf_ids(topo: Tuple[int, ...]) -> Iterator[str]:
    ranges = tuple(map(lambda t: range(1, t + 1), topo))
    for ids in product(*ranges):
        yield "tree." + ".".join(map(str, ids))


def read_ats_trace(
    infile: Path,
    vary_runtime: bool = True,
    vary_cores: bool = True,
    shuffle_seed: Optional[int] = None,
) -> Iterator[ExperimentJob]:
    infile = Path(infile)

    with infile.open("r", encoding="utf-8") as f:
        it = map(int, f)
        if shuffle_seed is not None:
            it = shuffle_iterable(it, shuffle_seed, n=1)

        for idx, core_count in enumerate(it):
            if vary_runtime and (core_count > 1):
                # maintain 1:6 ratio between runtimes for small and large jobs
                runtime = 720
                args = ["sleep", "60"]
            else:
                runtime = 120
                args = ["sleep", "10"]

            if not vary_cores:
                core_count = 1

            yield ExperimentJob(idx, args, core_count, runtime)


def distribute_rr(
    topo: Tuple[int, ...], jobs: Iterable[ExperimentJob]
) -> Dict[str, List[ExperimentJob]]:
    leaves = dict((leaf_id, []) for leaf_id in get_leaf_ids(topo))

    for j, leaf in zip(jobs, cycle(leaves.values())):
        leaf.append(j)

    return leaves


def distribute_by_cores(
    topo: Tuple[int, ...], jobs: Iterable[ExperimentJob]
) -> Dict[str, List[ExperimentJob]]:
    leaves = dict((leaf_id, []) for leaf_id in get_leaf_ids(topo))
    jobs = sorted(jobs, key=lambda j: j.cores)

    # Group together jobs based on core requirements, then round-robin distribute
    # jobs within groups among leaf schedulers, so that each group is (roughly)
    # equally distributed among each leaf.
    for _, group in groupby(jobs, key=lambda j: j.cores):
        for j, leaf in zip(group, cycle(leaves.values())):
            leaf.append(j)

    # Put jobs for each leaf back into workflow order
    for leaf in leaves.values():
        leaf.sort(key=lambda j: j.workflow_job_idx)

    return leaves


def distribute_by_utilization(
    topo: Tuple[int, ...], jobs: Iterable[ExperimentJob]
) -> Dict[str, List[ExperimentJob]]:
    ret = dict((leaf_id, []) for leaf_id in get_leaf_ids(topo))
    leaves = list(ret.values())
    total_utilization = sum(j.cores * j.timelimit for j in jobs)
    leaf_utilization = np.ceil(total_utilization / len(leaves))

    # prefix sum based load balancing:
    cur_sum = 0
    for j in jobs:
        assign_to = int(np.floor(cur_sum / leaf_utilization))
        leaves[assign_to].append(j)
        cur_sum += j.cores * j.timelimit

    return ret


def dump_distribution(outdir: Path, leaves: Dict[str, List[ExperimentJob]]):
    outdir = Path(outdir)

    for leaf_id, jobs in leaves.items():
        outfile = outdir.joinpath(leaf_id + ".json")
        with outfile.open("w", encoding="utf-8") as f:
            json.dump([j.as_dict() for j in jobs], f)


def parse_topology(s: str) -> Tuple[int, ...]:
    return tuple(map(int, s.split("x")))
