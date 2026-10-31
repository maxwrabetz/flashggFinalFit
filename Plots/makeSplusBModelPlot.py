#!/usr/bin/env python3
# Kept for the law tasks that call this path: runs scripts/plot_splusb.py (finalfits.splusb_plots)
import os
import runpy

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "plot_splusb.py"), run_name="__main__"
)
