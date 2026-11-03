"""Stage 1 -- gap detection and out-of-order handling."""

from laptop.acquire.seqtrack import SeqTracker


def feed(tracker, *batches):
    gaps = []
    for i, b in enumerate(batches):
        gaps += tracker.update(b, t_arrival_ns=i)
    return gaps


def test_clean_stream():
    t = SeqTracker()
    assert feed(t, range(0, 10), range(10, 20), range(20, 30)) == []
    assert (t.received, t.expected, t.received_frac) == (30, 30, 1.0)
    assert not t.missing and t.late == 0 and t.duplicates == 0


def test_stream_need_not_start_at_zero():
    t = SeqTracker()
    feed(t, range(500, 510))
    assert t.expected == 10 and t.received_frac == 1.0


def test_gap_is_reported_once_with_size_and_position():
    t = SeqTracker()
    gaps = feed(t, range(0, 10), range(25, 35))
    assert len(gaps) == 1
    assert (gaps[0].first_missing, gaps[0].count, gaps[0].last_missing) == (10, 15, 24)
    assert gaps[0].t_detected_ns == 1
    assert t.received == 20 and t.expected == 35 and len(t.missing) == 15
    assert t.received_frac == 20 / 35


def test_multiple_gaps():
    t = SeqTracker()
    gaps = feed(t, [0, 1, 2], [5, 6], [9, 10])
    assert [(g.first_missing, g.count) for g in gaps] == [(3, 2), (7, 2)]
    assert t.missing == {3, 4, 7, 8}


def test_out_of_order_batch_fills_the_gap():
    t = SeqTracker()
    gaps = feed(t, range(0, 10), range(20, 30), range(10, 20))  # 2nd and 3rd swapped
    assert len(gaps) == 1 and gaps[0].count == 10
    assert gaps[0].recovered == 10 and gaps[0].still_missing == 0
    assert t.late == 10 and not t.missing
    assert t.received == 30 and t.received_frac == 1.0


def test_out_of_order_within_a_batch():
    t = SeqTracker()
    feed(t, [0, 1, 3, 2, 4])
    assert t.late == 1 and not t.missing and t.received == 5


def test_partial_recovery():
    t = SeqTracker()
    gaps = feed(t, range(0, 5), range(15, 20), [7, 8])
    assert gaps[0].count == 10 and gaps[0].recovered == 2 and gaps[0].still_missing == 8
    assert len(t.missing) == 8


def test_duplicates_are_not_counted_twice():
    t = SeqTracker()
    feed(t, range(0, 10), range(5, 15))
    assert t.duplicates == 5 and t.received == 15 and t.received_frac == 1.0


def test_late_sample_before_first():
    t = SeqTracker()
    feed(t, [10, 11, 12], [8])
    assert t.first == 8 and t.missing == {9} and t.late == 1
    assert t.expected == 5 and t.received == 4


def test_lost_batch_at_99_9_percent():
    # 10 minutes at 100 Hz in batches of 10; lose 6 batches = 60 of 60000 = 0.1%
    t = SeqTracker()
    lost = {100, 900, 1800, 2700, 4000, 5900}
    for k in range(6000):
        if k not in lost:
            t.update(range(10 * k, 10 * k + 10), k)
    assert len(t.gaps) == 6
    assert t.received_frac == (60000 - 60) / 60000
