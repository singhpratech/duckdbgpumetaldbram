#!/usr/bin/env bash
# vram_sampler.sh — sample NVIDIA device memory while something else runs.
#
# The acceptance test for the memory-budget work is a shape, not a number:
# resident memory should PLATEAU at the budget rather than climb to the card's
# limit. Watching that needs a sample every few seconds across a long run,
# which is what this does.
#
#   scripts/vram_sampler.sh 6 600 > /tmp/vram.txt &   # every 6 s for 600 s
#   <run the gate, the wrapper suite, whatever>
#   awk '{print $2}' /tmp/vram.txt | sort -n | tail -1   # the peak
#
# Columns: unix_seconds used_MiB total_MiB. Exits quietly where nvidia-smi is
# absent, so it is safe to call unconditionally from a script.
set -u
every="${1:-6}"
for_seconds="${2:-600}"
command -v nvidia-smi >/dev/null 2>&1 || exit 0
end=$(( $(date +%s) + for_seconds ))
while [ "$(date +%s)" -lt "$end" ]; do
    printf '%s %s\n' "$(date +%s)" \
        "$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr ',' ' ')"
    sleep "$every"
done
