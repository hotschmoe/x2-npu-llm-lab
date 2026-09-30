import subprocess, sys
for s in ['hip_e0.py', 'hip_hvx.py', 'hip_e2.py', 'hip_e1.py']:
    print('#####', s, flush=True)
    r = subprocess.run([sys.executable, s], capture_output=True, text=True)
    print(r.stdout, r.stderr[-1500:], flush=True)
