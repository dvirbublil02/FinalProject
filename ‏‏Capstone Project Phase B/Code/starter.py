# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:23:17 2025

@author: dvirb
"""

import subprocess

# List of Python files to execute in order
scripts = [
    "visual.py", 
    "visualArtcles.py"
]

for script in scripts:
    print(f"Running {script}...")
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
