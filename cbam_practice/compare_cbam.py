"""CIFAR-10: identical small CNN with and without CBAM. Run from any directory."""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'vision_transformer_practice' / 'data'

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean((2, 3), keepdim=True)
        maximum = x.amax((2, 3), keepdim=True)
        channel = torch.sigmoid(self.mlp(avg) + self.mlp(maximum))
        x = x * channel
        spatial_input = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.spatial(spatial_input))

class Net(nn.Module):
    def __init__(self, attention):
        super().__init__()
        def block(in_channels, out_channels):
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.blocks = nn.ModuleList([block(3, 32), block(32, 64), block(64, 128)])
        self.head = nn.Linear(128, 10)
        # Construct shared layers before CBAM, so both models start from the same CNN parameters.
        self.attention = nn.ModuleList([CBAM(c) if attention else nn.Identity() for c in (32, 64, 128)])

    def forward(self, x):
        for block, attention in zip(self.blocks, self.attention):
            x = attention(block(x))
        return self.head(x.mean((2, 3)))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device):
    model.eval(); loss_sum = correct = count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x); loss_sum += nn.functional.cross_entropy(out, y, reduction='sum').item()
            correct += (out.argmax(1) == y).sum().item(); count += y.size(0)
    return {'loss': loss_sum / count, 'accuracy': correct / count, 'count': count}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for this experiment')
    device = torch.device('cuda')
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    plain = transforms.Compose([transforms.ToTensor(), normalize])
    train_data = datasets.CIFAR10(DATA, train=True, download=False, transform=aug)
    val_data = datasets.CIFAR10(DATA, train=True, download=False, transform=plain)
    test_data = datasets.CIFAR10(DATA, train=False, download=False, transform=plain)
    # Stratified split: 4,500 train and 500 validation examples for each class.
    rng = np.random.default_rng(args.seed); targets = np.asarray(train_data.targets)
    train_idx, val_idx = [], []
    for cls in range(10):
        idx = rng.permutation(np.flatnonzero(targets == cls)); val_idx.extend(idx[:500]); train_idx.extend(idx[500:])
    val_loader = DataLoader(Subset(val_data, val_idx), batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
    output = ROOT / 'results'; output.mkdir(exist_ok=True)
    results = {'config': {'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'train': len(train_idx), 'validation': len(val_idx), 'test': len(test_data),
              'optimizer': 'Adam', 'learning_rate': 0.001, 'device': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'split': 'stratified CIFAR-10 training set; seed 42 by default'}}
    for use_cbam in (False, True):
        name = 'cbam' if use_cbam else 'baseline'
        seed_all(args.seed); model = Net(use_cbam).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = []; best = None; best_val = -1; gradient_norms = None
        for epoch in range(1, args.epochs + 1):
            # Repeat shuffle and augmentation RNG for both arms each epoch.
            seed_all(args.seed + epoch)
            loader = DataLoader(Subset(train_data, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
            start = time.perf_counter(); model.train(); loss_sum = correct = count = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
                logits = model(x); loss = nn.functional.cross_entropy(logits, y); loss.backward()
                if use_cbam and gradient_norms is None:
                    gradient_norms = {'channel_mlp': model.attention[0].mlp[0].weight.grad.norm().item(),
                                      'spatial_conv': model.attention[0].spatial.weight.grad.norm().item()}
                optimizer.step(); loss_sum += loss.item()*y.size(0)
                correct += (logits.argmax(1)==y).sum().item(); count += y.size(0)
            val = evaluate(model, val_loader, device)
            record = {'epoch': epoch, 'train_loss': loss_sum/count, 'train_accuracy': correct/count,
                      'val_loss': val['loss'], 'val_accuracy': val['accuracy'], 'seconds': time.perf_counter()-start}
            history.append(record)
            print(name, json.dumps(record), flush=True)
            if val['accuracy'] > best_val:
                best_val = val['accuracy']; best = copy.deepcopy(model.state_dict()); best_epoch = epoch
        model.load_state_dict(best)
        test = evaluate(model, test_loader, device)
        torch.save(best, output / (name + '_best.pt'))
        results[name] = {'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
                         'history': history, 'test': test, 'first_batch_gradient_norms': gradient_norms}
        (output/'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        print(name, 'test', json.dumps(test), flush=True)
    print('Finished:', output/'results.json', flush=True)

if __name__ == '__main__': main()
