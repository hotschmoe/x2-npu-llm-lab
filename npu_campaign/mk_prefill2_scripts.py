BS = chr(92)
def w(src, dst, subs):
    s = open(src, encoding='utf-8').read()
    for a, b in subs:
        s = s.replace(a, b)
    open(dst, 'w', encoding='utf-8', newline='').write(s)

w('build-prefill-htp.cmd', 'build-prefill2-htp.cmd',
  [('llama.cpp-hexagon' + BS, 'llama.cpp-hexagon-prefill2' + BS), ('build-prefill-htp', 'build-prefill2-htp')])
w('prefill-cand.ps1', 'prefill2-cand.ps1',
  [(BS + 'llama.cpp-hexagon' + BS + '$Build', BS + 'llama.cpp-hexagon-prefill2' + BS + '$Build'),
   ('prefill-cand-', 'prefill2-cand-'), ("'build-prefill-htp'", "'build-prefill2-htp'")])
w('build-prefill-host.cmd', 'build-prefill2-host.cmd',
  [('llama.cpp-hexagon', 'llama.cpp-hexagon-prefill2'), ('build-prefill-host', 'build-prefill2-host')])
