#!/bin/bash
# Configure passwordless sudo for drop_caches (required by benchmark).
#
# This adds a NOPASSWD rule for the specific command used to drop the
# OS page cache. Only this one command is granted passwordless sudo.
#
# Usage (run once, as a user with sudo access):
#   bash scripts/setup_sudoers.sh
#
# What it does:
#   Adds to /etc/sudoers.d/drop_caches:
#     <user> ALL=(ALL) NOPASSWD: /usr/bin/sh -c sync && echo 3 > /proc/sys/vm/drop_caches
#
# Why it's safe:
#   - Scoped to a single harmless command (clears OS RAM cache, does NOT
#     delete files or data)
#   - Uses a dedicated sudoers.d file (easy to remove: sudo rm /etc/sudoers.d/drop_caches)
#   - Validated with visudo before applying

set -euo pipefail

SUDOERS_FILE="/etc/sudoers.d/drop_caches"
CURRENT_USER="${SUDO_USER:-$USER}"

# The exact command string that benchmark.py runs
RULE="${CURRENT_USER} ALL=(ALL) NOPASSWD: /usr/bin/sh -c sync && echo 3 > /proc/sys/vm/drop_caches"

echo "=== Setting up passwordless sudo for drop_caches ==="
echo "  User: $CURRENT_USER"
echo "  File: $SUDOERS_FILE"

# Write and validate the rule
echo "$RULE" | sudo tee "$SUDOERS_FILE" > /dev/null
sudo chmod 0440 "$SUDOERS_FILE"

# Validate with visudo to catch syntax errors
if sudo visudo -cf "$SUDOERS_FILE"; then
    echo "  OK: sudoers rule validated."
else
    echo "  ERROR: Invalid sudoers syntax. Removing file."
    sudo rm -f "$SUDOERS_FILE"
    exit 1
fi

# Quick smoke test
echo "=== Smoke test ==="
if sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; then
    echo "  OK: Page cache drop works without password prompt."
else
    echo "  WARNING: Page cache drop test failed."
    echo "  The benchmark will continue but epoch 0 may not be a true cold read."
    echo "  Check: sudo -l | grep drop_caches"
fi

echo ""
echo "Done. You can now run the benchmark without sudo prompts:"
echo "  python experiments/run_benchmark.py --epochs 3 --repeats 3"
echo ""
echo "To remove this rule later:"
echo "  sudo rm $SUDOERS_FILE"
