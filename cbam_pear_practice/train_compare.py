"""One-fold MobileNetV2 vs MobileNetV2+CBAM on DiaMOS pear leaves."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'cbam_practice'))
from compare_cbam import CBAM

CLASSES=['healthy','spot','curl','slug']

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class PearDataset(Dataset):
    def __init__(self, rows, transform): self.rows=rows; self.transform=transform
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(ROOT/row['cache_path']) as image:
            x=self.transform(image.convert('RGB'))
        return x, CLASSES.index(row['label'])

class PearNet(nn.Module):
    def __init__(self, with_cbam):
        super().__init__()
        self.features=mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.features.requires_grad_(False)
        # Initialize shared classifier before CBAM so both groups start identically.
        # Educational approximation: the exact hidden dimensions in the paper need verification.
        self.classifier=nn.Sequential(nn.BatchNorm1d(1280),nn.Linear(1280,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,4))
        self.attention=CBAM(1280) if with_cbam else nn.Identity()
    def train(self, mode=True):
        super().train(mode)
        self.features.eval()  # freeze running BN statistics in the backbone too
        return self
    def forward(self,x):
        with torch.no_grad(): x=self.features(x)
        x=self.attention(x)
        return self.classifier(x.mean(dim=(2,3)))

def evaluate(model, loader, device):
    model.eval(); losses=0; count=0; all_y=[]; all_pred=[]
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device)); y_gpu=y.to(device)
            losses+=nn.functional.cross_entropy(logits,y_gpu,reduction='sum').item()
            count+=len(y); all_y.extend(y.tolist()); all_pred.extend(logits.argmax(1).cpu().tolist())
    return {'loss':losses/count,'accuracy':accuracy_score(all_y,all_pred),
      'macro_f1':f1_score(all_y,all_pred,labels=list(range(4)),average='macro',zero_division=0),
      'confusion_matrix':confusion_matrix(all_y,all_pred,labels=list(range(4))).tolist(),
      'per_class':classification_report(all_y,all_pred,labels=list(range(4)),target_names=CLASSES,output_dict=True,zero_division=0),
      'count':count}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--patience',type=int,default=15)
    parser.add_argument('--min-epochs',type=int,default=25)
    parser.add_argument('--pilot',action='store_true',help='Run only the balanced 160-image exploratory pilot')
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU unavailable')
    manifest_path=ROOT/'data'/('pilot_256' if args.pilot else 'leaves_256')/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    rows=manifest['images']; y=np.array([CLASSES.index(row['label']) for row in rows])
    expected=160 if args.pilot else 3006
    if len(rows)!=expected: raise ValueError('Expected %d images, got %d'%(expected,len(rows)))
    splitter=StratifiedKFold(n_splits=10,shuffle=True,random_state=args.seed)
    train_val_idx,test_idx=next(splitter.split(np.zeros(len(y)),y))
    train_idx,val_idx=train_test_split(train_val_idx,test_size=0.1,random_state=args.seed,stratify=y[train_val_idx])
    assert len(set(train_idx)&set(val_idx))==len(set(train_idx)&set(test_idx))==len(set(val_idx)&set(test_idx))==0
    normal=transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    train_t=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),transforms.RandomRotation(15),transforms.ColorJitter(brightness=.1,contrast=.1,saturation=.1),transforms.ToTensor(),normal])
    eval_t=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normal])
    choose=lambda idx:[rows[int(i)] for i in idx]
    val_loader=DataLoader(PearDataset(choose(val_idx),eval_t),batch_size=args.batch_size,shuffle=False,num_workers=0)
    test_loader=DataLoader(PearDataset(choose(test_idx),eval_t),batch_size=args.batch_size,shuffle=False,num_workers=0)
    device=torch.device('cuda'); out=ROOT/('pilot_results' if args.pilot else 'results'); out.mkdir(exist_ok=True)
    split={'train':[int(i) for i in train_idx],'validation':[int(i) for i in val_idx],'test':[int(i) for i in test_idx]}
    (out/'split.json').write_text(json.dumps(split,indent=2),encoding='utf8')
    counts={k:{label:sum(y[idx]==i) for i,label in enumerate(CLASSES)} for k,idx in [('train',train_idx),('validation',val_idx),('test',test_idx)]}
    counts={k:{a:int(v) for a,v in v.items()} for k,v in counts.items()}
    results={'config':{'seed':args.seed,'max_epochs':args.epochs,'batch_size':args.batch_size,
        'learning_rate':2e-5,'optimizer':'Adam','patience':args.patience,'min_epochs':args.min_epochs,
        'split':'first fold of stratified 10-fold; validation split is 10% of remaining training pool',
        'class_counts':counts,'classes':CLASSES,'gpu':torch.cuda.get_device_name(0),
        'backbone':'torchvision MobileNetV2 ImageNet1K V1, frozen including BN statistics',
        'classifier':'BN(1280), Linear(1280,256), ReLU, Dropout(0.3), Linear(256,4)',
        'preprocessing':'original archive -> 256x256 JPEG cache -> 224x224, approximate paper',
        'caveat':('Balanced 160-image pilot, NOT paper-scale results; test contains 4 per class; ' if args.pilot else 'One fold; ')+ 'educational approximation; head and augmentation may differ from paper'}}
    print('split',json.dumps(counts),flush=True)
    for kind in ['baseline','cbam']:
        seed_all(args.seed); model=PearNet(kind=='cbam').to(device)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=2e-5)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5,min_lr=1e-6)
        history=[]; best_score=-1.; best_loss=float('inf'); stale=0; best=None; gradient=None; stop='maximum epochs'
        for epoch in range(1,args.epochs+1):
            seed_all(args.seed+epoch)
            loader=DataLoader(PearDataset(choose(train_idx),train_t),batch_size=args.batch_size,shuffle=True,num_workers=0)
            model.train(); total=correct=count=0; start=time.perf_counter()
            for x,y_batch in loader:
                x,y_batch=x.to(device),y_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x); loss=nn.functional.cross_entropy(logits,y_batch); loss.backward()
                if kind=='cbam' and gradient is None:
                    gradient={'channel_mlp':model.attention.mlp[0].weight.grad.norm().item(),
                              'spatial_conv':model.attention.spatial.weight.grad.norm().item()}
                optimizer.step(); total+=loss.item()*len(y_batch); count+=len(y_batch); correct+=(logits.argmax(1)==y_batch).sum().item()
            val=evaluate(model,val_loader,device)
            rec={'epoch':epoch,'train_loss':total/count,'train_accuracy':correct/count,'val_loss':val['loss'],
                 'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1'],'lr':optimizer.param_groups[0]['lr'],
                 'seconds':time.perf_counter()-start}
            history.append(rec); print(kind,json.dumps(rec),flush=True)
            if val['accuracy']>best_score:
                best_score=val['accuracy']; best_epoch=epoch; best=copy.deepcopy(model.state_dict())
                torch.save(best,out/(kind+'_best.pt'))
            if val['loss']<best_loss-1e-4: best_loss=val['loss']; stale=0
            else: stale+=1
            scheduler.step(val['loss'])
            results[kind]={'status':'training','history':history,'best_epoch':best_epoch,'best_val_accuracy':best_score}
            (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
            if epoch>=args.min_epochs and stale>=args.patience:
                stop='validation loss plateau'; break
        model.load_state_dict(best); test=evaluate(model,test_loader,device)
        results[kind]={'status':'complete','history':history,'best_epoch':best_epoch,'best_val_accuracy':best_score,
            'test':test,'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'gradient_norms_first_batch':gradient,'stop_reason':stop}
        (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
        print(kind,'test',json.dumps(test,ensure_ascii=False),flush=True)
        del model,optimizer,best
    print('Experiment completed:',out/'results.json',flush=True)

if __name__=='__main__': main()
