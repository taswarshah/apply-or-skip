#!/bin/bash
# EC2 user-data for Apply-or-Skip (Amazon Linux 2023, x86_64, >=16 GB RAM, CPU only).
# Installs the agent + strands-decider, runs the Decider on 127.0.0.1:8099 (never exposed),
# and installs the idle auto-stop timer. No secrets: Bedrock access comes from the instance role.
set -euxo pipefail
REPO_URL="${REPO_URL:-https://github.com/taswarshah/apply-or-skip.git}"
CHECKPOINT="${DECIDER_CHECKPOINT:-StrandsAgents/strands-decider-2B-hobson-v21}"
dnf install -y git python3.12 python3.12-pip
id aos 2>/dev/null || useradd -m aos
sudo -u aos bash -lc "
  set -e
  cd ~ && [ -d apply-or-skip ] || git clone '$REPO_URL' apply-or-skip
  cd apply-or-skip && python3.12 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -e '.[decider,dev]'
"
cat >/etc/systemd/system/strands-decider.service <<UNIT
[Unit]
Description=Strands Decider 2B (localhost only)
After=network-online.target
[Service]
User=aos
WorkingDirectory=/home/aos/apply-or-skip
ExecStart=/home/aos/apply-or-skip/.venv/bin/strands-decider serve $CHECKPOINT --host 127.0.0.1 --port 8099 --device cpu
Restart=on-failure
[Install]
WantedBy=multi-user.target
UNIT
install -m 0755 /home/aos/apply-or-skip/deploy/idle-stop.sh /usr/local/bin/aos-idle-stop
cat >/etc/systemd/system/aos-idle-stop.service <<UNIT
[Unit]
Description=Stop this instance when idle
[Service]
Type=oneshot
ExecStart=/usr/local/bin/aos-idle-stop
UNIT
cat >/etc/systemd/system/aos-idle-stop.timer <<UNIT
[Unit]
Description=Check for idleness every 5 minutes
[Timer]
OnBootSec=15min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now strands-decider.service aos-idle-stop.timer
