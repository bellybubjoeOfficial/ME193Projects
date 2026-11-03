#!/usr/bin/env bash
# Compile sketch/ and flash it to the UNO Q's MCU, from the board itself.
#
#   scripts/flash_sketch.sh
#
# The streamer is not an App Lab app, so App Lab won't flash this sketch for
# us. This does what App Lab does: arduino-cli compile with the sketch.yaml
# profile, then remoteocd (local OpenOCD) writes the Zephyr loader + sketch.
#
# Stop any running App Lab app first (it flashes its own sketch), and make
# sure no App Lab app is the boot default:
#   arduino-app-cli properties set default none
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
build="${BUILD_DIR:-$HOME/.cache/shin-imu/build}"
pkg="$HOME/.arduino15/packages/arduino"
core="$(ls -d "$pkg"/hardware/zephyr/* | sort -V | tail -1)"
remoteocd="$(ls -d "$pkg"/tools/remoteocd/* | sort -V | tail -1)/remoteocd"
loader="$core/firmwares/zephyr-arduino_uno_q_stm32u585xx.bin"
cfg="$core/variants/arduino_uno_q_stm32u585xx/flash_sketch.cfg"

if [[ "$(tr '\0' '\n' < /sys/firmware/devicetree/base/compatible | head -1)" != "arduino,imola" ]]; then
  echo "not an UNO Q (arduino,imola): refusing to flash" >&2
  exit 1
fi

if systemctl --user is-active --quiet shin-imu; then
  echo "note: shin-imu is running; it will reconnect to the MCU after the flash" >&2
fi

echo "== compiling ($core)"
arduino-cli compile --profile default --build-path "$build" "$here/sketch"

echo "== flashing"
"$remoteocd" upload -f "$cfg" "$loader" "$build/sketch.ino.elf-zsk.bin"
echo "== done"
