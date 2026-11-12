"""Compare a wider CNN against the sequential channel/spatial self-attention model."""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from compare_cbam import DATA, Net, evaluate, seed_all

ROOT = Path(__file__).resolve().parent

class SequentialSelfAttention(nn.Module):
    """Channel tokens first, then spatial tokens. Both use Q/K/V self-attention."""
    def __init__(self, channels, size, channel_dim=16):
        super().__init__()
        positions = size * size
        self.channel_qkv = nn.Linear(positions, 3 * channel_dim)
        self.channel_out = nn.Linear(channel_dim, positions)
        self.spatial_qkv = nn.Linear(channels, 3 * channels)
        self.spatial_out = nn.Linear(channels, channels)
        self.channel_scale = nn.Parameter(torch.tensor(0.1))
        self.spatial_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        b, c, h, w = x.shape
        # C tokens, each token encodes the full H*W spatial map.
        channel_tokens = x.reshape(b, c, h * w)
        q, k, v = self.channel_qkv(channel_tokens).chunk(3, dim=-1)
        mixed = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        x = x + self.channel_scale * self.channel_out(mixed).reshape(b, c, h, w)
        # H*W tokens, each token is a C-dimensional feature at one location.
        spatial_tokens = x.flatten(2).transpose(1, 2)
        q, k, v = self.spatial_qkv(spatial_tokens).chunk(3, dim=-1)
        mixed = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        mixed = self.spatial_out(mixed).transpose(1, 2).reshape(b, c, h, w)
        return x + self.spatial_scale * mixed


class WideNet(nn.Module):
    """Same three-block CNN, wider channels; 202,627 parameters."""
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.blocks = nn.ModuleList([block(3, 47), block(47, 94), block(94, 188)])
        self.head = nn.Linear(188, 10)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean((2, 3)))


def make_model(kind, seed):
    seed_all(seed)
    # The self-attention model retains the original CNN architecture.
    if kind == 'wide_cnn':
        return WideNet()
    model = Net(False)
    if kind == 'cbam':
        from compare_cbam import CBAM
        model.attention = nn.ModuleList([CBAM(c) for c in (32, 64, 128)])
    elif kind == 'self_attention':
        model.attention = nn.ModuleList([SequentialSelfAttention(c, s) for c, s in zip((32, 64, 128), (16, 8, 4))])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--min-epochs', type=int, default=20)
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    device = torch.device('cuda')
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    augment = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    plain = transforms.Compose([transforms.ToTensor(), norm])
    train = datasets.CIFAR10(DATA, train=True, download=False, transform=augment)
    val = datasets.CIFAR10(DATA, train=True, download=False, transform=plain)
    test = datasets.CIFAR10(DATA, train=False, download=False, transform=plain)
    rng = np.random.default_rng(args.seed); labels = np.asarray(train.targets)
    train_indices, val_indices = [], []
    for cls in range(10):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        val_indices.extend(idx[:500]); train_indices.extend(idx[500:])
    val_loader = DataLoader(Subset(val, val_indices), batch_size=256, shuffle=False)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)
    outdir = ROOT / 'convergence_results'
    outdir.mkdir(exist_ok=True)
    results = {'config': {'seed':args.seed, 'epochs':args.epochs, 'batch_size':args.batch_size,
        'train':len(train_indices), 'validation':len(val_indices), 'test':len(test),
        'optimizer':'Adam', 'learning_rate':0.001, 'lr_schedule':'ReduceLROnPlateau on validation loss, factor=0.5, patience=4, min_lr=1e-5', 'early_stopping':'validation loss, patience=12, minimum epochs=20', 'patience':args.patience, 'min_epochs':args.min_epochs, 'gpu':torch.cuda.get_device_name(0),
        'torch':torch.__version__, 'spatial_self_attention':'global, single-head',
        'channel_self_attention':'single-head, 16-dimensional Q/K/V', 'dtype':'float32'}}
    # These architectures have different widths; they cannot share identical initial tensors.
    for kind in ('wide_cnn','self_attention'):
        torch.cuda.empty_cache()
        model = make_model(kind,args.seed).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)
        torch.cuda.reset_peak_memory_stats(device)
        history=[]; best_val=-1; best=None; grads=None; best_loss=float('inf'); stale=0; stop_reason='maximum epochs'
        for epoch in range(1,args.epochs+1):
            seed_all(args.seed+epoch)
            loader=DataLoader(Subset(train,train_indices),batch_size=args.batch_size,shuffle=True,num_workers=0)
            torch.cuda.synchronize(); start=time.perf_counter()
            model.train(); total=correct=count=0
            for x,y in loader:
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x); loss=F.cross_entropy(logits,y); loss.backward()
                if kind=='self_attention' and grads is None:
                    layer=model.attention[0]
                    grads={'channel_qkv':layer.channel_qkv.weight.grad.norm().item(),
                           'spatial_qkv':layer.spatial_qkv.weight.grad.norm().item()}
                optimizer.step()
                total+=loss.item()*y.size(0); correct+=(logits.argmax(1)==y).sum().item(); count+=y.size(0)
            validation=evaluate(model,val_loader,device)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            record={'epoch':epoch,'train_loss':total/count,'train_accuracy':correct/count,
                    'val_loss':validation['loss'],'val_accuracy':validation['accuracy'],'seconds':elapsed, 'learning_rate':optimizer.param_groups[0]['lr']}
            history.append(record); print(kind,json.dumps(record),flush=True)
            if validation['accuracy']>best_val:
                best_val=validation['accuracy']; best=copy.deepcopy(model.state_dict()); best_epoch=epoch
                torch.save(best,outdir/(kind+'_best.pt'))
            if validation['loss'] < best_loss - 0.0001:
                best_loss = validation['loss']; stale = 0
            else:
                stale += 1
            scheduler.step(validation['loss'])
            results[kind] = {'parameters':sum(p.numel() for p in model.parameters()), 'best_epoch':best_epoch,
                             'history':history, 'status':'training', 'best_val_loss':best_loss}
            (outdir/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
            if epoch >= args.min_epochs and stale >= args.patience:
                stop_reason = 'validation loss did not improve for %d epochs' % args.patience
                print(kind, 'early stop:', stop_reason, flush=True)
                break
        peak=torch.cuda.max_memory_allocated(device)
        model.load_state_dict(best); test_result=evaluate(model,test_loader,device)
        torch.save(best,outdir/(kind+'_best.pt'))
        results[kind]={'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,
            'history':history,'test':test_result,'status':'complete','stop_reason':stop_reason,'best_val_loss':best_loss,'peak_allocated_mib_training_and_validation':peak/1024**2,
            'first_batch_gradient_norms':grads}
        (outdir/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
        print(kind,'test',json.dumps(test_result),'peak MiB',peak/1024**2,flush=True)
        del model,optimizer,best
    print('Finished',outdir/'results.json',flush=True)

if __name__=='__main__': main()
