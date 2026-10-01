# lockhold.py -- cmd...: like npu_campaign\lockrun.py (same lock dir/protocol, NPU_LOCK_HELD=1) but polls every 20 ms so a long-lived
# server can win the lock against back-to-back benchmark holders; owner label "serve". Releases only a lock it owns.
import os, sys, time, subprocess, shutil
lock = r'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\.npu.lock'
cmd = sys.argv[1:]
if cmd and cmd[0] == '--': cmd = cmd[1:]
mine = f'serve lockhold pid={os.getpid()}'
while True:
    try:
        os.mkdir(lock); open(os.path.join(lock, 'owner.txt'), 'w').write(f'{mine} started={time.ctime()}'); break
    except FileExistsError:
        try:
            f = os.path.join(lock, 'owner.txt'); pid = int(open(f).read().split('pid=')[1].split()[0])
            alive = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True).stdout.count(str(pid)) > 0
            if not alive and time.time() - os.path.getmtime(f) > 120: shutil.rmtree(lock, ignore_errors=True)
        except Exception: pass
        time.sleep(0.02)
try:
    env = dict(os.environ); env['NPU_LOCK_HELD'] = '1'
    sys.exit(subprocess.run(cmd, env=env).returncode)
finally:
    try:
        if mine in open(os.path.join(lock, 'owner.txt')).read(): shutil.rmtree(lock, ignore_errors=True)
    except Exception: pass
