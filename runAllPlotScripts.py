import subprocess
import os

for script in os.listdir("correlationJointPlots"):
    if script.endswith(".py"):
        subprocess.run(f"python correlationJointPlots/{script}", shell=True)