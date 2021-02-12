import numpy as np

from sched_model import (
    System,
    Job,
    fcfs,
    easy_backfill,
    conservative_backfill,
    hybrid_backfill,
)


def display_system(system: System):
    util_vecs = []
    cur_jobs = []

    for t, node in system.iter_timeline():
        s = ""

        if len(node.start) > 0:
            reserved = list(filter(lambda j: j.is_reserved, node.start))
            running = list(filter(lambda j: not j.is_reserved, node.start))

            if len(reserved) > 0:
                s += "reserved job{} {}".format(
                    "s" if len(reserved) > 1 else "",
                    ", ".join(map(lambda j: str(j.job_id), reserved)),
                )

            if len(running) > 0:
                if len(s) > 0:
                    s += " / "

                s += "started job{} {}".format(
                    "s" if len(running) > 1 else "",
                    ", ".join(map(lambda j: str(j.job_id), running)),
                )

            for j in node.start:
                cur_jobs.append(j)

        if len(node.expired) > 0:
            if len(s) > 0:
                s += " / "

            s += "job{} {} expire{}".format(
                "s" if len(node.expired) > 1 else "",
                ", ".join(map(lambda j: str(j.job_id), node.expired)),
                "s" if len(node.expired) == 1 else "",
            )

            for j in node.expired:
                cur_jobs.remove(j)

        ended_jobs = node.end - node.expired
        if len(ended_jobs) > 0:
            if len(s) > 0:
                s += " / "

            s += "finished job{} {}".format(
                "s" if len(ended_jobs) > 1 else "",
                ", ".join(map(lambda j: str(j.job_id), ended_jobs)),
            )

            for j in ended_jobs:
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

            prefix = "{:>2d} |".format(t)
            if t == system.cur_time:
                prefix = "> " + prefix
            else:
                prefix = "  " + prefix

            print(prefix + util_bar, end=end)

        prev_tm = tm
        prev_jobs = job_list
        prev_msg = msg

    if system.cur_time == tm:
        s = "> {:>2d} |".format(system.cur_time)
        for max_util in system.total_resources:
            s += ("." * max_util) + "|"
        s += " (workflow complete)"
        print(s)


if __name__ == "__main__":
    system = System(np.array([6]))

    system.enqueue_job(Job(2, np.array([1])))
    system.enqueue_job(Job(3, np.array([1])))
    system.enqueue_job(Job(5, np.array([2])))
    system.enqueue_job(Job(4, np.array([6])))
    system.enqueue_job(Job(3, np.array([1])))
    system.enqueue_job(Job(5, np.array([2])))
    system.enqueue_job(Job(1, np.array([3])))
    system.enqueue_job(Job(2, np.array([4])))
    system.enqueue_job(Job(1, np.array([1])))
    sched_policy = easy_backfill

    i = 1
    while True:
        system.run_sched_loop(sched_policy)

        print("Scheduler loop {} (t={}):".format(i, system.cur_time))
        display_system(system)
        print("\n")
        i += 1

        if not system.handle_events():
            break

