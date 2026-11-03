"""Stage 3 -- matching detections to hand-marked truth."""

from laptop.vision.compare import compare, match_events


def test_perfect_match():
    pairs, missed, extra = match_events([10, 40, 70], [10, 40, 70], tol=6)
    assert pairs == [(10, 10), (40, 40), (70, 70)] and missed == [] and extra == []


def test_missed_extra_and_out_of_tolerance():
    pairs, missed, extra = match_events([12, 55, 90], [10, 40, 70], tol=6)
    assert pairs == [(12, 10)]
    assert missed == [40, 70] and extra == [55, 90]


def test_one_to_one_closest_first():
    # Two detections near one true strike: only the closest matches.
    pairs, missed, extra = match_events([38, 41], [40], tol=6)
    assert pairs == [(41, 40)] and extra == [38] and missed == []


def test_compare_per_leg_and_totals():
    det = [(10, "L"), (41, "R"), (72, "L"), (100, "R")]
    truth = [(9, "L"), (40, "R"), (70, "L"), (130, "R")]
    r = compare(det, truth, tol=6)
    assert r["L"]["errors"] == [1, 2]
    assert r["R"]["missed"] == [130] and r["R"]["extra"] == [100]
    t = r["total"]
    assert (t["detected"], t["truth"], t["count_error"]) == (4, 4, 0)
    assert (t["matched"], t["missed"], t["extra"]) == (3, 1, 1)
    assert t["median_abs_err"] == 1.0 and t["max_abs_err"] == 2


def test_legs_never_cross_match():
    r = compare([(10, "L")], [(10, "R")], tol=6)
    assert r["total"]["matched"] == 0 and r["total"]["missed"] == 1 and r["total"]["extra"] == 1
