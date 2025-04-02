#!/usr/bin/env python
"""
Run ECG Simulator - Launcher script for the ECG Viewer with simulation capabilities
"""
import sys
import os

# Set VERSION and LOG_LEVEL globally to avoid circular import
VERSION = "2.0 with Simulator"
LOG_LEVEL = "INFO"

# Make these variables available to imported modules
os.environ["ECG_VIEWER_VERSION"] = VERSION
os.environ["ECG_VIEWER_LOG_LEVEL"] = LOG_LEVEL

# Import and run our simulator
from ecg_viewer_simulator import main

if __name__ == "__main__":
    main() 