# interleaved A/B: MoE expert matmuls via HMX dequant pipeline (ID_HVX_MAX_T=4, default) vs forced HVX Q4_0 x Q8_0 int8 vrmpy path (ID_HVX_MAX_T=4096)
import os, re, subprocess, sys
ROOT = r'C:\Users\hotschmoe\Documents\llm_inference'
pkg = os.path.join(ROOT, 'npu_campaign', 'pkg-merged2-m1')
exe = os.path.join(pkg, 'bin', 'llama-bench.exe')
model = os.path.join(ROOT, 'models', 'gguf', 'qwen36-35b-a3b-q4_0', 'Qwen_Qwen3.6-35B-A3B-Q4_0.gguf')
lockrun = os.path.join(ROOT, 'npu_campaign', 'int4u8_lockrun.py')
pp, ub, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for r in range(rounds):
    for val in ('4', '4096'):
        env = dict(os.environ)
        env.update(GGML_HEXAGON_OPBATCH='3072', GGML_HEXAGON_REPLAY='1', GGML_HEXAGON_OPFUSION='1', GGML_HEXAGON_ROUTER_HVX='1', GGML_HEXAGON_MM_SELECT='2',
                   GGML_HEXAGON_FA_SELECT='2', GGML_HEXAGON_GDN_SELECT='2', GGML_HEXAGON_ID_HVX_MAX_T=val)
        env['ADSP_LIBRARY_PATH'] = os.path.join(pkg, 'lib') + ';'
        cmd = [sys.executable, lockrun, '--', exe, '-m', model, '-p', pp, '-n', '0', '-ub', ub, '-b', '2048', '-r', '1', '-ngl', '99', '--device', 'HTP0', '-o', 'md']
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
        m = re.search(r'pp\d+\s*\|\s*([\d.]+)', p.stdout)
        tps = float(m.group(1)) if m else None
        print('round', r + 1, 'ID_HVX_MAX_T=' + val, 'pp' + pp, 'ub' + ub, '->', tps, 't/s', flush=True)
        if not m: print(p.stdout[-800:], p.stderr[-800:], flush=True)
