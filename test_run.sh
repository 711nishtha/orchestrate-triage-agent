#!/bin/bash
# test_run.sh: Quick test loop for the support agent

# 1. Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment .venv not found. Please run: python3 -m venv .venv && .venv/bin/pip install -r code/requirements.txt"
    exit 1
fi

# 2. Run the agent
echo "Running support agent..."
python3 code/main.py "$@"

# 3. Final status
echo "Done. Check support_tickets/support_tickets/output.csv and log file at \$HOME/hackerrank_orchestrate/log.txt"
