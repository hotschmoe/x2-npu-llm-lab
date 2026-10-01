"""One-model-at-a-time guard (48 GB machine). Every 2 s: if more than one llama-*.exe model process exists, kill all but the OLDEST by exact PID
(the newest is the one that bypassed the lock). Logs to npu_campaign/ram-guard.log. Never matches by command line, only by image name llama-*.exe.
Run: python npu_campaign/ram-guard.py   (keep it running in the background)"""
import csv, io, os, subprocess, time, datetime

NAMES = ('llama-cli.exe', 'llama-bench.exe', 'llama-server.exe', 'llama-completion.exe', 'llama-perplexity.exe', 'llama-batched-bench.exe', 'llama-mtmd-cli.exe')
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ram-guard.log')
SELF = os.getpid()
KILLED = set()

def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def model_procs():
    # (pid, image, start) for llama-* processes; start from PowerShell CIM (creation time) so "oldest" is well defined
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'llama-*.exe' } | "
          "ForEach-Object { '{0},{1},{2},{3}' -f $_.ProcessId, $_.Name, $_.CreationDate.ToString('o'), $_.WorkingSetSize }")
    out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True).stdout
    procs = []
    for row in csv.reader(io.StringIO(out)):
        # skip PIDs we already killed and dying zombies (a live model process holds GBs resident; a zombie only a few MB)
        if len(row) == 4 and row[1].lower() in NAMES and int(row[0]) not in KILLED and int(row[3]) > 200 * 1024 * 1024:
            procs.append((int(row[0]), row[1], row[2]))
    return sorted(procs, key=lambda p: p[2])

log(f"ram-guard started (pid {SELF})")
while True:
    try:
        procs = model_procs()
        if len(procs) > 1:
            keep = procs[0]
            for pid, name, start in procs[1:]:
                KILLED.add(pid)
                r = subprocess.run(['taskkill', '/PID', str(pid), '/F', '/T'], capture_output=True, text=True)
                log(f"KILLED second model process pid={pid} {name} start={start} (keeping pid={keep[0]} {keep[1]}): {r.stdout.strip() or r.stderr.strip()}")
    except Exception as e:  # never die
        log(f"guard error: {e!r}")
    time.sleep(2)
