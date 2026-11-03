"""python -m streamer --config config/streamer.toml"""

import argparse
import logging
import signal
import sys

from . import config
from .app import Streamer


def main():
    ap = argparse.ArgumentParser(description="Shin IMU -> MQTT streamer (UNO Q Linux side)")
    ap.add_argument("--config", required=True, help="TOML config file")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = config.load(args.config)
    s = Streamer(cfg)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: s.stop_event.set())
    s.run()


if __name__ == "__main__":
    main()
