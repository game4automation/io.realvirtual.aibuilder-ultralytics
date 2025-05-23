### Pyinstaller Bundling Guide

for bundling ultralytics, the source code needs to be adepted
in ultralytics trainer.py, comment the line calling print_args
for execution, in the cfg/default.yaml, set workers=0 
