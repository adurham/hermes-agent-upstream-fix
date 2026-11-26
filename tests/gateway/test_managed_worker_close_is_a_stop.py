"""A managed worker's own teardown is a STOP, not a lost worker, and the loss record can name
the signal that killed a child.

The owner closes a worker's pipe itself (a cancelled spawn, the turn's own ``finally``), and
``EOFError`` on that pipe was booked as a crash because ``close()`` never latched the Stop the
read loop converts into ``managed_worker_stopped`` (t_a9b941db). And a loss only ever said
``EOFError``: the child's wait status — SIGKILL vs exit 0 vs our own SIGTERM — was held by the
owner at that instant and thrown away, which is why 81 losses in one day could not be named.
"""
import asyncio
import subprocess
import sys

import pytest

from persistence.core.errors import AuthorityRefusal
from tests.waiting import async_wait_until

# Says nothing and never exits on its own: the reader parks on the pipe.
QUIET = "import time\ntime.sleep(300)\n"
# Exits at once without a word: the reader gets a real EOF from a child nobody closed.
EXITS = "import sys\nsys.exit(0)\n"


def _spawn(script):
    return subprocess.Popen([sys.executable, '-c', script], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


@pytest.mark.platforms("linux")
def test_a_closed_worker_reads_as_a_stop_not_a_lost_pipe():
    from gateway.session_managed_worker import ManagedWorker
    child = _spawn(QUIET)
    worker = ManagedWorker(child)

    async def scenario():
        reading = asyncio.create_task(worker.next_frame(None, ack=5))
        await async_wait_until(lambda: worker._requested, what="the reader parked on the pipe")
        await asyncio.to_thread(worker.close)  # our own teardown, from a worker thread
        return await asyncio.wait_for(reading, timeout=30)

    try:
        with pytest.raises(AuthorityRefusal) as stopped:
            asyncio.run(scenario())
    finally:
        worker.close()
    assert stopped.value.reason == 'managed_worker_stopped', stopped.value
    assert worker.stop.is_set() and worker.closed.is_set()
    assert child.poll() is not None


@pytest.mark.platforms("linux")
def test_an_eof_nobody_closed_is_still_a_loss():
    """The guard on the fix: a child that dies on its own must not be laundered into a Stop."""
    from gateway.session_managed_worker import ManagedWorker
    child = _spawn(EXITS)
    worker = ManagedWorker(child)
    try:
        async def scenario():
            return await asyncio.wait_for(worker.next_frame(None, ack=5), timeout=30)
        with pytest.raises(EOFError):
            asyncio.run(scenario())
    finally:
        worker.close()


@pytest.mark.platforms("linux")
def test_the_loss_state_names_the_signal_that_killed_the_child():
    from gateway.session_managed_worker import ManagedWorker
    child = _spawn(QUIET)
    worker = ManagedWorker(child)
    try:
        child.kill()  # SIGKILL: the census the board needs to be able to read
        assert worker.loss_state() == {'exit_status': -9, 'owner_closed': False}
        assert child.returncode == -9
    finally:
        worker.close()


@pytest.mark.platforms("linux")
def test_the_loss_state_reports_our_own_close():
    from gateway.session_managed_worker import ManagedWorker
    child = _spawn(QUIET)
    worker = ManagedWorker(child)
    worker.close()  # SIGTERM, and the latch must say it was ours
    assert worker.loss_state() == {'exit_status': -15, 'owner_closed': True}


@pytest.mark.platforms("linux")
def test_the_loss_state_reports_a_clean_exit_as_zero():
    from gateway.session_managed_worker import ManagedWorker
    child = _spawn(EXITS)
    worker = ManagedWorker(child)
    try:
        assert worker.loss_state(wait=10) == {'exit_status': 0, 'owner_closed': False}
    finally:
        worker.close()
