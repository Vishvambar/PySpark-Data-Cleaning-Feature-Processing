#!/usr/bin/env bash
# ==============================================================================
# PySpark Environment Launcher & Pipeline Execution Wrapper
# Automatically binds to OpenJDK 17 and Python 3.12 venv.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Environment Configuration
JDK_DIR="$HOME/.local/share/jvm/jdk-17.0.12+7"
VENV_DIR="$HOME/.local/share/pyspark-venv"

if [ -d "$JDK_DIR" ]; then
    export JAVA_HOME="$JDK_DIR"
    export PATH="$JDK_DIR/bin:$PATH"
fi

if [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON_EXEC="$VENV_DIR/bin/python"
    export PYSPARK_PYTHON="$PYTHON_EXEC"
    export PYSPARK_DRIVER_PYTHON="$PYTHON_EXEC"
else
    PYTHON_EXEC="$(which python3)"
fi

# 2. Execution Routing
if [ "$1" == "test" ] || [ "$1" == "--test" ]; then
    echo "Running PySpark Automated Verification Suite..."
    "$PYTHON_EXEC" run_all_operations.py
elif [ "$1" == "pipeline" ] || [ "$1" == "--pipeline" ]; then
    echo "Running PySpark End-to-End Pipeline..."
    "$PYTHON_EXEC" pyspark_pipeline.py
elif [ "$1" == "vis" ] || [ "$1" == "--vis" ]; then
    echo "Generating High-Resolution Visualizations..."
    "$PYTHON_EXEC" visualize_features.py
else
    "$PYTHON_EXEC" main.py
fi
