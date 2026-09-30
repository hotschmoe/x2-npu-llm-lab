# ops650: lockrun.py with 10 ms lock polling and the stale-lock check only every 5 s (the shared version polls every 0.5 s and loses to 200 ms pollers).
import os, sys, time, subprocess, shutil
lock = r'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\.npu.lock'
cmd = sys.argv[1:]
if cmd and cmd[0] == '--': cmd = cmd[1:]
last_check = 0.0
while True:
    try:
        os.mkdir(lock); open(os.path.join(lock, 'owner.txt'), 'w').write(f'ops650 lockrun pid={os.getpid()} started={time.ctime()}'); break
    except FileExistsError:
        if time.time() - last_check > 5:
            last_check = time.time()
            try:
                f = os.path.join(lock, 'owner.txt'); pid = int(open(f).read().split('pid=')[1].split()[0])
                alive = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True).stdout.count(str(pid)) > 0
                if not alive and time.time() - os.path.getmtime(f) > 120: shutil.rmtree(lock, ignore_errors=True)
            except Exception: pass
        time.sleep(0.01)
try:
    env = dict(os.environ); env['NPU_LOCK_HELD'] = '1'
    sys.exit(subprocess.run(cmd, env=env).returncode)
finally:
    shutil.rmtree(lock, ignore_errors=True)
