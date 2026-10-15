"""Idle work between requests: it starts a delay after the last request, one step at a time, and yields to a request."""

import threading
import time

from tensorfold.cuda.idle_work import IdleWork


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.005)


def test_idle_work_runs_its_steps_after_the_delay():
    ran = []
    work = IdleWork(lambda: [lambda i=i: ran.append(i) for i in range(3)], delay=0.05)
    with work.busy():
        pass
    started = time.monotonic()
    wait_for(lambda: ran == [0, 1, 2])
    assert time.monotonic() - started >= 0.05
    work.close()


def test_a_request_stops_the_steps_and_they_start_over_after_it():
    ran, gate = [], threading.Event()

    def slow(i):
        ran.append(i)
        gate.wait(2)

    plans = []

    def plan():
        plans.append(len(ran))
        return [lambda i=i: slow(i) for i in range(len(ran), 5)]

    work = IdleWork(plan, delay=0.01)
    with work.busy():
        pass
    wait_for(lambda: ran == [0])                    # a step is running
    done = threading.Event()

    def request():
        with work.busy():                            # waits for the running step, then holds the GPU
            assert len(ran) == 1                     # no step started while it waited
        done.set()

    t = threading.Thread(target=request)
    t.start()
    time.sleep(0.05)
    gate.set()                                       # the running step finishes; the request goes first
    wait_for(done.is_set)
    wait_for(lambda: ran == [0, 1, 2, 3, 4])         # after the request, a new plan picks up the rest
    assert plans[:2] == [0, 1]
    work.close()


def test_no_plan_no_work():
    work = IdleWork(lambda: None, delay=0.01)
    with work.busy():
        pass
    time.sleep(0.05)
    assert work.steps == 0
    work.close()
