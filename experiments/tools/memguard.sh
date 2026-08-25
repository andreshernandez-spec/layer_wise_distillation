#!/bin/bash
# Run a command and kill it if its process tree exceeds CAP_GB of RSS or if the
# machine's MemAvailable drops below FLOOR_GB. Userspace substitute for a cgroup cap
# (the user slice on this laptop does not enforce MemoryMax). Poll every 2 s.
#   experiments/tools/memguard.sh CAP_GB FLOOR_GB -- command args...
set -u
CAP=$1; FLOOR=$2; shift 3
"$@" &
PID=$!
tree_rss_kb() { ps -o rss= --ppid "$1" -p "$1" 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
while kill -0 $PID 2>/dev/null; do
  rss=$(tree_rss_kb $PID)
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$rss" -gt $((CAP * 1024 * 1024)) ] || [ "$avail" -lt $((FLOOR * 1024 * 1024)) ]; then
    echo "MEMGUARD: killing pid $PID (tree rss $((rss/1024)) MB, avail $((avail/1024)) MB)" >&2
    pkill -TERM -P $PID; kill -TERM $PID; sleep 3; pkill -KILL -P $PID 2>/dev/null; kill -KILL $PID 2>/dev/null
    exit 137
  fi
  sleep 2
done
wait $PID
