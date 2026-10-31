#!/usr/bin/env python3
# Kept for the law tasks that call this path: runs scripts/correct_impacts.py (finalfits.impacts)
import os
import runpy

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "correct_impacts.py"), run_name="__main__"
)
