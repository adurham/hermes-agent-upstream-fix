"""The control writer is dormant until there is a control to write, and never overtakes a bootstrap.

Starting it with the worker cost a wake-every-0.5 s thread for the whole life of every worker, and
most workers are never controlled at all: 15 live workers held 15 of them in a gateway already at
233 threads, where the cost of scheduling them is what starves the event loop. Order is the other
half of the contract -- a Stop queued before the bootstrap frame went down stdin must not reach the
child ahead of it, which is why the writer is ARMED at the bootstrap boundary and started by the
first control frame queued at or after that.
"""
import asyncio
import subprocess
import sys

import pytest

# Echoes the type of every frame it receives, in order, so the test can prove what arrived and when.
ECHO = """import json, sys
for line in sys.stdin.buffer:
    sys.stdout.buffer.write(json.dumps({'type': 'delta', 'text': json.loads(line)['type']}).encode() + b'\\n')
    sys.stdout.buffer.flush()
"""


def _worker():
    from gateway.session_managed_worker import ManagedWorker
    child = subprocess.Popen([sys.executable, '-c', ECHO], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return ManagedWorker(child)


def _read(worker, count):
    async def read():
        frames = []
        async with asyncio.timeout(15):
            while len(frames) < count:
                frames.append(await worker.next_frame(None, ack=5))
        return [f['text'] for f in frames]
    return asyncio.run(read())


@pytest.mark.platforms("linux")
def test_arming_with_nothing_to_write_leaves_the_writer_dormant():
    """A worker that is never controlled never gets the thread: arming alone starts nothing."""
    worker = _worker()
    try:
        assert worker.writer.ident is None
        worker.arm_writer()
        assert worker.writer.ident is None          # armed, but the queue was empty
        assert not worker.writer.is_alive()

        worker.control({'type': 'stop'})            # the first control starts it
        assert worker.writer.ident is not None

        assert _read(worker, 1) == ['stop']
    finally:
        worker.close()


@pytest.mark.platforms("linux")
def test_a_stop_queued_before_the_bootstrap_waits_for_the_bootstrap():
    """Queue a Stop while the child is still being bootstrapped: it must not reach stdin first."""
    worker = _worker()
    try:
        worker.control({'type': 'stop'})            # unarmed: queued, and provably not written
        assert worker.writer.ident is None
        assert not worker.writer.is_alive()

        worker.send({'type': 'bootstrap'})          # the frame that must arrive first
        worker.arm_writer()                         # now the queued Stop may go
        assert worker.writer.ident is not None

        assert _read(worker, 2) == ['bootstrap', 'stop']
    finally:
        worker.close()
