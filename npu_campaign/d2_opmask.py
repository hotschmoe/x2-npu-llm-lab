# d2_opmask.py NAME1,NAME2,... -> "GGML_HEXAGON_DIAG_AFTER_MASK_LO=0x..u;..._HI=..;..._X=.." for HTP_OP_<NAME>
import re, sys
src = open(r'C:\Users\hotschmoe\documents\llm_inference\llama.cpp-hexagon-decode2\ggml\src\ggml-hexagon\htp\htp-ops.h').read()
body = src[src.index('HTP_OP_MUL = 0'):src.index('HTP_OP_INVALID')]
names = []; val = 0; mp = {}
for line in body.split('\n'):
    line = re.sub(r'//.*', '', line).strip().rstrip(',')
    m = re.match(r'HTP_OP_(\w+)(?:\s*=\s*(\d+))?$', line)
    if m:
        if m.group(2): val = int(m.group(2))
        mp[m.group(1)] = val; val += 1
arg = sys.argv[1]
inv = arg.startswith('~')
sel = [mp[n] for n in arg.lstrip('~').split(',')]
m = [0, 0, 0]
for o in sel: m[o // 32] |= 1 << (o % 32)
if inv: m = [~x & 0xffffffff for x in m]
print(f"GGML_HEXAGON_DIAG_AFTER_MASK_LO=0x{m[0]:x}u;GGML_HEXAGON_DIAG_AFTER_MASK_HI=0x{m[1]:x}u;GGML_HEXAGON_DIAG_AFTER_MASK_X=0x{m[2]:x}u")
