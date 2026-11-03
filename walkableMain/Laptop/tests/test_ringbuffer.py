"""Stage 1 -- ring buffer wrap-around, ordering, and thread safety."""

import threading

import numpy as np

from laptop.acquire.ringbuffer import RingBuffer


def add(rb, seqs):
    seqs = np.asarray(seqs, dtype=np.int64)
    rb.append(seqs * 10, seqs, np.tile(seqs[:, None], (1, 6)).astype(float))


def test_fill_and_wrap_keeps_newest():
    rb = RingBuffer(25)
    for k in range(10):
        add(rb, range(10 * k, 10 * k + 10))
    t, seq, data = rb.snapshot()
    assert len(rb) == 25
    np.testing.assert_array_equal(seq, np.arange(75, 100))
    np.testing.assert_array_equal(data[:, 3], np.arange(75, 100))


def test_batch_bigger_than_capacity():
    rb = RingBuffer(8)
    add(rb, range(20))
    np.testing.assert_array_equal(rb.snapshot()[1], np.arange(12, 20))


def test_snapshot_sorted_and_windowed():
    rb = RingBuffer(100)
    add(rb, [0, 1, 2, 5, 6])
    add(rb, [3, 4])  # late arrivals
    t, seq, _ = rb.snapshot()
    np.testing.assert_array_equal(seq, np.arange(7))
    t, seq, _ = rb.snapshot(since_ns=40)
    np.testing.assert_array_equal(seq, [4, 5, 6])


def test_concurrent_writer_and_reader():
    rb = RingBuffer(500)
    stop = threading.Event()
    errors = []

    def reader():
        while not stop.is_set():
            t, seq, data = rb.snapshot()
            if len(seq) and not np.array_equal(data[:, 0], seq):
                errors.append("torn read")

    th = threading.Thread(target=reader)
    th.start()
    for k in range(2000):
        add(rb, range(10 * k, 10 * k + 10))
    stop.set()
    th.join()
    assert not errors
