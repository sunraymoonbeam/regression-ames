#!/bin/bash
set -euo pipefail

# Train the full pipeline and write Kaggle submissions.
# Override any config value on the command line, e.g.:
#   ./run.sh training.models=[ridge,lasso] training.n_iter=20
python main.py "$@"
