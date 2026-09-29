import sys, hashlib, struct
import numpy as np
path, n = sys.argv[1], int(sys.argv[2])
rows = []
with open(path, 'rb') as f:
    while h := f.read(20):
        magic, ctx, row, cnt = struct.unpack('<IQiI', h)
        pl = f.read(cnt * 8)
        v = np.frombuffer(pl, dtype='<u4').reshape(cnt, 2)[:, 1].copy().view('<f4')
        nn = int(np.isnan(v).sum())
        rows.append((row, hashlib.sha256(pl).hexdigest()[:10] + ('!NaN%d' % nn if nn else '') + ':top%d' % (int(np.nanargmax(v)) if nn < cnt else -1)))
per = len(rows) // n
bad = 0
for r in range(per):
    hs = [rows[i * per + r][1] for i in range(n)]
    bad += len(set(hs)) > 1
    print(f'row {r}: distinct={len(set(hs))} ' + ' '.join(hs))
req = [tuple(rows[i * per + r][1] for r in range(per)) for i in range(n)]
print(f'RESULT n={n} rows_per_req={per} rows_with_variation={bad} distinct_request_traces={len(set(req))}')
