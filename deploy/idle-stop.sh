#!/bin/bash
# Powers off the instance after IDLE_MINUTES with no SSM/SSH session and low CPU.
# Instance must be launched with --instance-initiated-shutdown-behavior stop (NOT terminate).
IDLE_MINUTES="${IDLE_MINUTES:-30}"
STAMP=/var/tmp/aos-last-active
CPU_IDLE_PCT=$(top -bn2 -d1 | awk '/^%Cpu/{v=$8} END{print int(v)}')
SESSIONS=$( (who | wc -l; pgrep -fc 'ssm-session-worker' || true) | paste -sd+ | bc)
if [ "${SESSIONS:-0}" -gt 0 ] || [ "${CPU_IDLE_PCT:-100}" -lt 90 ]; then
  date +%s > "$STAMP"; exit 0
fi
[ -f "$STAMP" ] || date +%s > "$STAMP"
IDLE=$(( ( $(date +%s) - $(cat "$STAMP") ) / 60 ))
if [ "$IDLE" -ge "$IDLE_MINUTES" ]; then
  logger "aos-idle-stop: idle ${IDLE} min, powering off"
  shutdown -h now
fi
