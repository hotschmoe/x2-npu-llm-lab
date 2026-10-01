# qwen4-guard.py -- cmd args...: pre-flight for ANY model launch (run INSIDE the NPU lock hold): wait until (a) available RAM > 20000 MB and (b) no live llama-* process exists,
# then run the command and return its exit code. Exits 97 if the conditions do not hold within 15 minutes (the caller must not start a model then).
import ctypes, subprocess, sys, time

class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

def avail_mb():
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys // (1024 * 1024)

def live_llama():
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "(Get-Process llama-* -ErrorAction SilentlyContinue | Where-Object { -not $_.HasExited } | ForEach-Object { $_.Id }) -join ','"],
                         capture_output=True, text=True).stdout.strip()
    return out

cmd = sys.argv[1:]
if cmd and cmd[0] == '--': cmd = cmd[1:]
t0 = time.time()
while True:
    a = avail_mb(); l = live_llama()
    if a > 20000 and not l: break
    if time.time() - t0 > 900:
        print(f'qwen4-guard: refusing to start: available {a} MB, live llama-* pids [{l}]', file=sys.stderr); sys.exit(97)
    time.sleep(5)
sys.exit(subprocess.run(cmd).returncode)
