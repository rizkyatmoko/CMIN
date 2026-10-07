"""Per-seed spread of the full adjacency A and its learned part A_ctx, by rung."""
import numpy as np, io, glob, os, collections, re

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final'))
L = lambda f: np.load(io.BytesIO(open(f, 'rb').read()), allow_pickle=True)
fs = sorted(glob.glob('extract/*.npz.zip'))
print(collections.Counter(re.sub(r'__s\d+\.npz\.zip$', '', os.path.basename(f)) for f in fs))

for r in ('L6_ctx_conditioned@S', 'L6_ctx_conditioned@R', 'L7_ctx_shuffled@S', 'UNTRAINED_L6@S',
          'L3_learned_static@R', 'H_uniform_price@R', 'L6nofilm@R'):
    g = sorted(glob.glob('extract/%s__s*.npz.zip' % r))
    if not g:
        print(r, 'none')
        continue
    x = L(g[0])
    print(r, len(g), {k: x[k].shape for k in x.files})
    rows = []
    for f in g:
        x = L(f)
        if 'Actx_mean' in x.files:
            rows.append((round(float(x['A_mean'].std()), 4), float('%.2g' % x['Actx_mean'].std())))
    print('   (A std, Actx std) per seed:', rows)
