import subprocess, sys
for s in ['hip_s3a.py', 'hip_s3b.py']:
    print('#####', s, flush=True)
    r = subprocess.run([sys.executable, s], capture_output=True, text=True)
    print('\n'.join(l for l in r.stdout.splitlines() if ' rc ' not in l), r.stderr[-1500:], flush=True)
