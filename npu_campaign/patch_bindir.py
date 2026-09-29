import os
here = os.path.dirname(os.path.abspath(__file__))
BS = chr(92)
for f, exe in [('run-prefill-bench.ps1', 'llama-bench.exe'), ('run-prefill-ppl.ps1', 'llama-perplexity.exe')]:
    p = os.path.join(here, f)
    s = open(p, encoding='utf-8').read()
    if 'BinDir' in s:
        continue
    s = s.replace("[string]$LibDir = ''", "[string]$LibDir = '',\n    [string]$BinDir = ''", 1)
    old = '$exe = "$repoRoot' + BS + 'llama.cpp-hexagon' + BS + 'pkg-wos' + BS + 'bin' + BS + exe + '"'
    assert old in s, f
    new = ('$binDirResolved = if ($BinDir) { (Resolve-Path (Join-Path $repoRoot $BinDir)).Path } else { "$repoRoot' + BS +
           'llama.cpp-hexagon' + BS + 'pkg-wos' + BS + 'bin" }\n$exe = "$binDirResolved' + BS + exe + '"')
    s = s.replace(old, new, 1)
    open(p, 'w', encoding='utf-8').write(s)
    print('ok', f)
