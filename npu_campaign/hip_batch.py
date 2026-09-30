# runs test-hmx-int-probe once per job file (crash isolation), 60 s timeout each; invoked under lockrun.py
import subprocess, sys
exe = sys.argv[1]
for p in sys.argv[2:]:
    try:
        r = subprocess.run([exe, p], capture_output=True, text=True, timeout=90)
        print(p, 'rc', r.returncode, (r.stdout.strip().splitlines() or [''])[-1], flush=True)
    except subprocess.TimeoutExpired:
        print(p, 'TIMEOUT', flush=True)
