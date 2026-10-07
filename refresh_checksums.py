"""Rewrite CHECKSUMS.sha256 for every file in this package.

    python refresh_checksums.py
"""
import hashlib, os

EV = os.path.dirname(os.path.abspath(__file__))
lines, total = [], 0
for root, _, files in os.walk(EV):
    for f in sorted(files):
        if f == 'CHECKSUMS.sha256':
            continue
        p = os.path.join(root, f)
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        rel = os.path.relpath(p, EV).replace(os.sep, '/')
        lines.append('%s  %s' % (h.hexdigest(), rel))
        total += os.path.getsize(p)
open(os.path.join(EV, 'CHECKSUMS.sha256'), 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
print('CHECKSUMS.sha256: %d files, %.0f MB' % (len(lines), total / 1048576))
