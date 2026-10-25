import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def evaluate(model, dataloader, device):
    model.eval()
    preds, targets, features = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y, d = batch
            x = x.to(device)
            out = model(x)
            y_pred = out[0] if isinstance(out, tuple) else out
            z = out[-1] if isinstance(out, tuple) else x.reshape(x.size(0), -1)
            
            preds.extend(y_pred.argmax(dim=1).cpu().numpy())
            targets.extend(y.numpy())
            features.extend(z.cpu().numpy())
    
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    return dict(acc=acc, f1=f1, y=np.array(targets), pred=np.array(preds), z=np.array(features))

def run_experiment(ds, method='baseline', lam=1.0, seed=0, epochs=30, num_classes=6, device='cuda'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from torch.utils.data import DataLoader
    from .models import CNNLSTM, DANN
    
    tr_loader = DataLoader(ds['src_tr'], batch_size=32, shuffle=True)
    va_loader = DataLoader(ds['src_va'], batch_size=32, shuffle=False)
    te_loader = DataLoader(ds['tgt_te'], batch_size=32, shuffle=False)
    
    if method == 'baseline':
        model = CNNLSTM(num_classes).to(device)
    else:
        model = DANN(num_classes).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_dom = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_acc': [], 'target_acc': [], 'dom_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y, d in tr_loader:
            x, y, d = x.to(device), y.to(device), d.to(device)
            optimizer.zero_grad()
            
            if method == 'baseline':
                out, _ = model(x)
                loss = criterion_cls(out, y)
            else:
                y_pred, d_pred, _ = model(x, alpha=lam)
                loss_cls = criterion_cls(y_pred, y)
                loss_dom = criterion_dom(d_pred, d.long())
                loss = loss_cls + lam * loss_dom
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        history['train_loss'].append(epoch_loss / len(tr_loader))
        
        rs = evaluate(model, va_loader, device)
        rt = evaluate(model, te_loader, device)
        history['val_acc'].append(rs['acc'])
        history['target_acc'].append(rt['acc'])
        
    rs = evaluate(model, va_loader, device)
    rt = evaluate(model, te_loader, device)
    return model, history, rs, rt
