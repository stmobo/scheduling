import numpy as np

from sched_model import (
    System,
    Job,
    fcfs,
    easy_backfill,
    conservative_backfill,
    hybrid_backfill,
)

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
