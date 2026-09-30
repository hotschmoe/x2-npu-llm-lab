import subprocess, sys, os
os.environ['NPU_LOCK_HELD'] = '1'
for cmd in (['hip_s4a.py'],):
    print('#####', cmd, flush=True)
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    print('\n'.join(l for l in r.stdout.splitlines() if ' rc ' not in l), r.stderr[-1500:], flush=True)
