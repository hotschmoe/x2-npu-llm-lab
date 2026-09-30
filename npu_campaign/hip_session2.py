import subprocess, sys
for s in ['hip_s2a.py', 'hip_s2d.py', 'hip_s2c.py', 'hip_s2b.py']:
    print('#####', s, flush=True)
    r = subprocess.run([sys.executable, s], capture_output=True, text=True)
    print(r.stdout, r.stderr[-1500:], flush=True)
