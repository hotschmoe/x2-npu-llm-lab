import subprocess, sys
for s in ['hip_s4b.py', 'hip_s4a.py']:
    print('#####', s, flush=True)
    r = subprocess.run([sys.executable, s], capture_output=True, text=True)
    print('\n'.join(l for l in r.stdout.splitlines() if ' rc ' not in l), r.stderr[-2500:], flush=True)
