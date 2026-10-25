"""Training + evaluasi satu konfigurasi. Contoh:
python run.py --root /path/BVP --method dann --lam 1.0 --seed 0"""
import argparse, json, os, torch
from src.dataset import make_datasets
from src.train import run_experiment
from src.utils import plot_curves, plot_cm, plot_tsne

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--method', choices=['baseline', 'dann'], default='dann')
ap.add_argument('--lam', type=float, default=1.0)
ap.add_argument('--lowpass', action='store_true')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--out', default='results')
a = ap.parse_args()

cfg = json.load(open('splits/config.json'))
nc, names = len(cfg['GESTURES']), [f'G{g}' for g in cfg['GESTURES']]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tag = f"{a.method}_lam{a.lam}_lp{int(a.lowpass)}_s{a.seed}"
os.makedirs(a.out, exist_ok=True)

ds = make_datasets(a.root, 'splits', lowpass=a.lowpass)
model, hist, rs, rt = run_experiment(ds, a.method, lam=a.lam, seed=a.seed,
                                     epochs=a.epochs, num_classes=nc, device=dev)
torch.save(model.state_dict(), f'{a.out}/{tag}.pt')
json.dump(hist, open(f'{a.out}/{tag}_hist.json', 'w'))
json.dump(dict(target_acc=rt['acc'], target_f1=rt['f1'], source_val_acc=rs['acc']),
          open(f'{a.out}/{tag}_metrics.json', 'w'), indent=2)
plot_curves(hist, tag, f'{a.out}/{tag}_curves.png')
plot_cm(rt['y'], rt['pred'], names, tag, f'{a.out}/{tag}_cm.png')
plot_tsne(rs['z'], rt['z'], tag, f'{a.out}/{tag}_tsne.png')
print(f"{tag} | TARGET acc {rt['acc']:.4f} | F1 {rt['f1']:.4f} | source val acc {rs['acc']:.4f}")
