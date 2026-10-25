## Proyek Pengganti UTS Mata Kuliah Pembelajaran Mesin Lanjut
 **Anggota Kelompok 12:**
  1. Prima Azahra       — NIM: 2411511005
  2. Shahida Shintya    — NIM: 2411511007
  3. Zega Vibiano Delta — NIM: 2411511009
  4. Fathia Muhammad Martha - NIM: 2211513057

---
## Struktur Direktori Proyek
* `BVP/` — Dataset mentah file MATLAB (.mat) per user
* `splits/` — File CSV indeks pembagian dataset (`src_tr.csv`, `src_va.csv`, `tgt_tr.csv`, `tgt_te.csv`)
* `results/` — Folder output checkpoint model (`.pt`) & grafik hasil visualisasi (`fig_tsne.png`)
* `Notebook 1 dan 2.ipynb` — Preprocessing dan Baseline CNN-LSTM
* `Notebook 3 dan 4.ipynb` — Training DANN, Evaluasi, dan t-SNE Analysis
* `README.md` — Dokumentasi resmi proyek
  
### Persyaratan Sistem & Dependensi
Proyek ini dijalankan pada lingkungan Google Colab dengan runtime Python 3.x dan akselerasi GPU (CUDA/T4). Packages utama yang digunakan:
1. torch & torchvision (Deep Learning Framework)
2. scikit-learn (Evaluasi metrik & reduksi dimensi t-SNE)
3. scipy (Pembacaan format file MATLAB .mat)
4. pandas & numpy (Manipulasi data & array)
5. matplotlib & seaborn (Visualisasi data & grafik)

### Panduan Eksekusi Program
- Persiapan Google Drive & Environment, pertama pastikan folder proyek telah diunggah ke Google Drive pada path:/content/drive/MyDrive/Project_UTS_BPV, lalu pastikan struktur folder BVP dan splits berada di dalam direktori utama proyek.

- Menjalankan Model & Evaluasi t-SNE
Jalankan sel eksekusi t-SNE berikut di Notebook 4 untuk mengekstraksi fitur laten dan memvisualisasikan domain adaptation:
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from google.colab import drive

### 1. Mount Google Drive
drive.mount('/content/drive', exist_ok=True)

### 2. Setup Path Utama
BASE_DIR = '/content/drive/MyDrive/Project_UTS_BPV'             
RESULTS = os.path.join(BASE_DIR, 'results')                 
SEED = 42                          
tu = users[-1]  # Target user

### 3. Load Datasets & Models
f = make_datasets(root=BASE_DIR, split_dir='splits')                     
base = BaselineCNNLSTM().to(device)                                      
base.load_state_dict(torch.load(f"{RESULTS}/baseline_target_user{tu}.pt", map_location=device))                                        
dann = DANN().to(device)                                
dann.load_state_dict(torch.load(f"{RESULTS}/dann_lam1.0_target_user{tu}.pt", map_location=device))

#### 4. Plot Visualisasi t-SNE
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))                                  
for ax, m, title in [(axes[0], base, "Baseline"), (axes[1], dann, "DANN")]:                    
    fs, _ = extract_features(m, f["val"], device)                                   
    ft, _ = extract_features(m, f["target"], device)                                               
    z = TSNE(n_components=2, perplexity=min(30, (len(fs) + len(ft) - 1) // 3),                                                          
        random_state=SEED).fit_transform(np.vstack([fs, ft]))                                              
    ax.scatter(z[:len(fs), 0], z[:len(fs), 1], s=10, c="tab:blue", alpha=.6, label="Source")                                                          
    ax.scatter(z[len(fs):, 0], z[len(fs):, 1], s=10, c="tab:red", alpha=.6, label=f"Target (user{tu})")                                                             
    ax.set_title(f"t-SNE Fitur - {title}")                                      
    ax.legend()                                             
plt.tight_layout()                                                       
plt.savefig(f"{RESULTS}/fig_tsne.png", dpi=300)                                              
plt.show()

### Hasil & Analisis
Visualisasi t-SNE menunjukkan perbandingan sebaran fitur antar domain:            
a) Baseline CNN-LSTM: Terdapat pemisahan (clustering gap) yang cukup berjarak antara data Source dan Target, mengindikasikan adanya pergeseran domain.           
b) DANN (Domain-Adversarial Neural Network): Distribusi fitur Source dan Target saling berhimpit (overlap), membuktikan bahwa gradient reversal layer (GRL) berhasil menyelaraskan ruang fitur domain secara simultan.



