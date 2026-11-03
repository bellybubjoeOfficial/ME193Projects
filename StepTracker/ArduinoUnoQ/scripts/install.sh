#!/usr/bin/env bash
# Install the streamer as a user systemd service that starts at boot.
#
#   scripts/install.sh            venv + unit + enable + start
#
# Does not flash the MCU (scripts/flash_sketch.sh) and does not touch App Lab.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
venv="$HOME/.venvs/shin-imu"

if [[ "$here" != "$HOME/shin-step-counter/device" ]]; then
  echo "warning: the unit expects the repo at ~/shin-step-counter/device (found $here)" >&2
fi

if [[ ! -x "$venv/bin/python" ]]; then
  echo "== creating $venv"
  python3 -m venv --system-site-packages "$venv"
fi
"$venv/bin/python" -c "import paho.mqtt, msgpack" 2>/dev/null || {
  echo "== installing requirements"
  "$venv/bin/python" -m pip install -r "$here/requirements.txt"
}

mkdir -p "$HOME/.config/systemd/user" "$HOME/.config/shin-imu"
install -m 644 "$here/systemd/shin-imu.service" "$HOME/.config/systemd/user/shin-imu.service"

# Without lingering, user services only run while someone is logged in.
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]]; then
  loginctl enable-linger "$USER" || echo "warning: run 'sudo loginctl enable-linger $USER' so the service starts at boot" >&2
fi

systemctl --user daemon-reload
systemctl --user enable --now shin-imu
systemctl --user --no-pager status shin-imu | head -5
