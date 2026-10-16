"""
logging_utils.py
Lightweight logging and telemetry for SU(8 radiance framework.
"""

import time

class PipelineLogger:
    def __init__(self):
        self.events = []

    def log(self, message: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {message}"
        print(entry)
        self.events.append(entry)

    def summary(self):
        print("\n=== Pipeline Log Summary ===")
        for e in self.events:
            print(e)
        print("=== End Summary ===")
