# lockrun.py [--] cmd args...  : take the NPU lock with 10 ms polling (fair against back-to-back holders), run the command with NPU_LOCK_HELD=1, release.
import os, sys, time, subprocess, shutil
lock = r'C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\.npu.lock'
cmd = sys.argv[1:]
if cmd and cmd[0] == '--': cmd = cmd[1:]
while True:
    try:
        os.mkdir(lock); open(os.path.join(lock, 'owner.txt'), 'w').write(f'decode2 lockrun pid={os.getpid()} started={time.ctime()}'); break
    except FileExistsError: time.sleep(0.01)
try:
    env = dict(os.environ); env['NPU_LOCK_HELD'] = '1'
    sys.exit(subprocess.run(cmd, env=env).returncode)
finally:
    shutil.rmtree(lock, ignore_errors=True)
