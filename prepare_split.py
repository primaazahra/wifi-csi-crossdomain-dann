"""Scan BVP -> pilih subset seimbang -> split source/target -> simpan ke splits/."""
import argparse, os, re, glob, json
import pandas as pd
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--domain', choices=['user', 'torso'], default='user')
ap.add_argument('--gestures', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
ap.add_argument('--n_src', type=int, default=3)
ap.add_argument('--n_tgt', type=int, default=2)
ap.add_argument('--source', type=int, nargs='*')     
ap.add_argument('--target', type=int, nargs='*')
ap.add_argument('--min_per_cell', type=int, default=10)
ap.add_argument('--per_cell', type=int, default=40)
ap.add_argument('--no_fix_others', action='store_true')
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()

files = glob.glob(os.path.join(a.root, '**', '*.mat'), recursive=True)
pat = re.compile(r'user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)')
rows = []
for f in files:
    m = pat.search(os.path.basename(f))
    if m:
        u, g, t, fo, r = map(int, m.groups())
        rows.append(dict(rel=os.path.relpath(f, a.root), user=u, gesture=g, torso=t, face=fo, rep=r))
df = pd.DataFrame(rows)
print('Total .mat:', len(files), '| nama file terparse:', len(df))
if df.empty:
    raise SystemExit('Tidak ada file terparse. Cek --root dan format nama file.')
print('\nuser x gesture\n', pd.crosstab(df.user, df.gesture))
print('\ntorso x gesture\n', pd.crosstab(df.torso, df.gesture))

D = a.domain
base = df[df.gesture.isin(a.gestures)].copy()
if not a.no_fix_others:
    for c in (['torso', 'face'] if D == 'user' else ['face']):
        v = base[c].mode()[0]
        print(f'Faktor {c} ditahan konstan = {v}')
        base = base[base[c] == v]

cnt = (base.groupby([D, 'gesture']).size().unstack(fill_value=0)
           .reindex(columns=a.gestures, fill_value=0))
ok = cnt[(cnt >= a.min_per_cell).all(axis=1)]
ok = ok.assign(total=ok.sum(axis=1)).sort_values('total', ascending=False)
print('\nDomain yang lolos (semua gestur >= min_per_cell):\n', ok)

if a.source and a.target:
    source, target = a.source, a.target
else:
    cands = [int(x) for x in ok.index]
    if len(cands) < a.n_src + a.n_tgt:
        raise SystemExit('Domain lolos kurang. Turunkan --min_per_cell atau pakai --no_fix_others.')
    source, target = cands[:a.n_src], cands[a.n_src:a.n_src + a.n_tgt]
print(f'\nSOURCE ({D}):', source, '| TARGET:', target)

parts = []
for (d, g), grp in base[base[D].isin(source + target)].groupby([D, 'gesture']):
    parts.append(grp.sample(min(len(grp), a.per_cell), random_state=a.seed))
sel = pd.concat(parts)
sel['label'] = sel.gesture.map({g: i for i, g in enumerate(a.gestures)})

src, tgt = sel[sel[D].isin(source)], sel[sel[D].isin(target)]
src_tr, src_va = train_test_split(src, test_size=0.2, stratify=src.label, random_state=a.seed)
tgt_tr, tgt_te = train_test_split(tgt, test_size=0.3, stratify=tgt.label, random_state=a.seed)

os.makedirs('splits', exist_ok=True)
for name, d in [('src_tr', src_tr), ('src_va', src_va), ('tgt_tr', tgt_tr), ('tgt_te', tgt_te)]:
    d.to_csv(f'splits/{name}.csv', index=False)
    print(f'{name}: {len(d)} sampel')
json.dump(dict(domain=D, source=source, target=target, GESTURES=a.gestures,
               per_cell=a.per_cell, seed=a.seed),
          open('splits/config.json', 'w'), indent=2)
print('Split tersimpan di splits/')
