"""U4.1 -- UNO Q MQTT payload parsing + staleness (no broker needed)."""

import json
import unittest

from unoq_link import SensorCache, parse_message, parse_status


def dist(seq, mm, t=1000.0):
    return json.dumps({"seq": seq, "t": t, "mm": mm})


class TestParseDistance(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_message("distance", dist(3, 423))["mm"], 423)

    def test_null_mm_is_valid(self):
        self.assertIsNone(parse_message("distance", dist(3, None))["mm"])

    def test_rejects_malformed(self):
        bad = [
            "not json",
            "[1, 2]",
            json.dumps({"t": 1.0, "mm": 5}),                  # no seq
            json.dumps({"seq": "3", "t": 1.0, "mm": 5}),      # seq as string
            json.dumps({"seq": True, "t": 1.0, "mm": 5}),     # bool isn't int
            json.dumps({"seq": 3, "mm": 5}),                  # no t
            json.dumps({"seq": 3, "t": 1.0}),                 # no mm key
            json.dumps({"seq": 3, "t": 1.0, "mm": -1}),       # negative
            json.dumps({"seq": 3, "t": 1.0, "mm": 4.5}),      # float mm
            json.dumps({"seq": 3, "t": 1.0, "mm": "far"}),
        ]
        for payload in bad:
            with self.assertRaises(ValueError, msg=payload):
                parse_message("distance", payload)


class TestParseImu(unittest.TestCase):
    def test_valid(self):
        p = json.dumps({"seq": 1, "t": 1.0, "ax": 0, "ay": 0.1, "az": 1.0,
                        "gx": 0, "gy": 0, "gz": 12.5})
        self.assertEqual(parse_message("imu", p)["gz"], 12.5)

    def test_missing_axis(self):
        p = json.dumps({"seq": 1, "t": 1.0, "ax": 0, "ay": 0, "az": 1})
        with self.assertRaises(ValueError):
            parse_message("imu", p)


class TestParseStatus(unittest.TestCase):
    def test_states(self):
        self.assertEqual(parse_status('{"state": "online", "sensors": ["distance"]}'), "online")
        self.assertEqual(parse_status('{"state": "offline"}'), "offline")
        self.assertIsNone(parse_status("garbage"))
        self.assertIsNone(parse_status('{"nope": 1}'))


class TestSensorCache(unittest.TestCase):
    def setUp(self):
        self.c = SensorCache(stale_s=0.3)

    def test_nothing_received_is_unknown(self):
        self.assertIsNone(self.c.distance_mm(0.0))

    def test_fresh_reading(self):
        self.c.on_message("distance", dist(0, 250), now=10.0)
        self.assertEqual(self.c.distance_mm(10.1), 250)

    def test_goes_stale_after_0_3_s(self):
        self.c.on_message("distance", dist(0, 250), now=10.0)
        self.assertEqual(self.c.distance_mm(10.29), 250)
        self.assertIsNone(self.c.distance_mm(10.31))

    def test_out_of_range_is_unknown_not_far(self):
        self.c.on_message("distance", dist(0, None), now=10.0)
        self.assertIsNone(self.c.distance_mm(10.0))

    def test_bad_message_dropped_and_keeps_last_good(self):
        self.c.on_message("distance", dist(0, 250), now=10.0)
        self.assertIsNone(self.c.on_message("distance", "garbage", now=10.05))
        self.assertEqual(self.c.distance_mm(10.1), 250)
        self.assertEqual(self.c.stats("distance", 10.1)["dropped"], 1)

    def test_bad_message_does_not_refresh_staleness(self):
        self.c.on_message("distance", dist(0, 250), now=10.0)
        self.c.on_message("distance", "garbage", now=10.25)
        self.assertIsNone(self.c.distance_mm(10.35))

    def test_seq_gaps_counted_as_missed(self):
        for seq, t in ((0, 0.0), (1, 0.05), (4, 0.1)):   # 2 and 3 lost
            self.c.on_message("distance", dist(seq, 100), now=t)
        self.assertEqual(self.c.stats("distance", 0.1)["missed"], 2)

    def test_publisher_restart_is_not_a_gap(self):
        self.c.on_message("distance", dist(500, 100), now=0.0)
        self.c.on_message("distance", dist(0, 100), now=0.05)
        self.assertEqual(self.c.stats("distance", 0.05)["missed"], 0)
        self.assertEqual(self.c.distance_mm(0.05), 100)

    def test_rate(self):
        for i in range(21):                      # 21 msgs over 1 s = 20 Hz
            self.c.on_message("distance", dist(i, 100), now=i * 0.05)
        self.assertAlmostEqual(self.c.stats("distance", 1.0)["rate_hz"], 20.0)

    def test_sensors_are_independent(self):
        self.c.on_message("distance", dist(0, 250), now=10.0)
        self.assertIsNone(self.c.latest("imu", 10.0))


if __name__ == "__main__":
    unittest.main()
