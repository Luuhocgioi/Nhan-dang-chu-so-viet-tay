"""Train/validation tách từ EMNIST train; test chính thức chỉ đo sau chọn model."""
import argparse
from datetime import datetime
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import EMNIST
from torchvision.transforms import RandomAffine
from sklearn.metrics import confusion_matrix,classification_report
from .config import ROOT,save_json
from .emnist_model import CharacterCNN,CHARACTERS,normalize_glyph


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--per-class',type=int,default=2500)
    args=parser.parse_args()
    if args.epochs<1 or args.per_class<1:parser.error('epochs và per-class phải dương.')
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    source=EMNIST(str(ROOT/'data/raw/emnist'),split='byclass',train=True,download=False)
    rng=np.random.default_rng(42)
    labels=source.targets.numpy()
    train_indices=[];val_indices=[]
    for c in range(62):
        ids=rng.permutation(np.flatnonzero(labels==c))
        val_indices.extend(ids[:200]);train_indices.extend(ids[200:200+args.per_class])
    def prepare(indices):
        x=np.stack([normalize_glyph(source.data[int(i)].numpy().T) for i in indices])
        return TensorDataset(torch.from_numpy(x[:,None]).float()/255,torch.tensor(labels[indices],dtype=torch.long))
    train=prepare(train_indices);val=prepare(val_indices)
    run='emnist_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    directory=ROOT/'models'/run;directory.mkdir(parents=True)
    report=ROOT/'reports/experiments'/run;report.mkdir(parents=True)
    save_json(report/'config.json',dict(vars(args),seed=42,device=device,train_samples=len(train),val_samples=len(val),
        normalization='transpose raw EMNIST; tight crop; preserve ratio to20; center in28',characters=CHARACTERS,
        torch=torch.__version__,source='NIST EMNIST ByClass',digit_threshold=.65))
    np.savez_compressed(report/'split_indices.npz',train=train_indices,validation=val_indices)
    loaders={k:DataLoader(ds,batch_size=256,shuffle=k=='train',num_workers=0) for k,ds in [('train',train),('val',val)]}
    model=CharacterCNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
    affine=RandomAffine(12,translate=(.08,.08),scale=(.85,1.15),shear=6)
    history=[];best=float('inf');started=time.perf_counter()
    for epoch in range(args.epochs):
        row={'epoch':epoch+1}
        for phase,loader in loaders.items():
            model.train(phase=='train');loss_sum=correct=count=0
            for x,y in loader:
                x,y=x.to(device),y.to(device)
                if phase=='train':x=affine(x)
                with torch.set_grad_enabled(phase=='train'):
                    logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y)
                    if phase=='train':
                        opt.zero_grad();loss.backward();opt.step()
                loss_sum+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();count+=len(y)
            row[phase+'_loss']=loss_sum/count;row[phase+'_accuracy']=correct/count
        scheduler.step();history.append(row);save_json(report/'history.json',history)
        print(row,flush=True)
        if row['val_loss']<best:
            best=row['val_loss'];torch.save(model.state_dict(),directory/'best.pt')
    model.load_state_dict(torch.load(directory/'best.pt',map_location=device,weights_only=True));model.eval()
    # Chỉ sau khi chọn checkpoint bằng val mới đọc test.
    test=EMNIST(str(ROOT/'data/raw/emnist'),split='byclass',train=False,download=False)
    predicted=[];truth=[];digit_fp=letter_count=digits=correct_digits=0
    with torch.inference_mode():
        for start in range(0,len(test),512):
            x=np.stack([normalize_glyph(a.numpy().T) for a in test.data[start:start+512]])
            p=model(torch.from_numpy(x[:,None]).float().to(device)/255).softmax(1).cpu().numpy()
            y=test.targets[start:start+512].numpy();pred=p.argmax(1)
            accept=(pred<10)&(p.max(1)>=.65)
            digit_fp+=int(((y>=10)&accept).sum());letter_count+=int((y>=10).sum())
            digits+=int((y<10).sum());correct_digits+=int(((y<10)&accept&(pred==y)).sum())
            predicted.extend(pred.tolist());truth.extend(y.tolist())
    metrics={'character_accuracy':float(np.mean(np.array(predicted)==truth)),
        'digit_correct_and_accepted':correct_digits,'digit_targets':digits,'digit_recall':correct_digits/digits,
        'letters_as_digits':digit_fp,'letter_targets':letter_count,'letter_false_positive_rate':digit_fp/letter_count,
        'threshold':.65,'seconds':time.perf_counter()-started,'test_samples':len(test),
        'confusion_matrix':confusion_matrix(truth,predicted,labels=range(62)).tolist(),
        'classification_report':classification_report(truth,predicted,labels=range(62),target_names=list(CHARACTERS),output_dict=True,zero_division=0)}
    save_json(report/'metrics.json',metrics)
    save_json(ROOT/'models/emnist_candidate.json',{'model':(directory/'best.pt').relative_to(ROOT).as_posix(),'run_id':run})
    print('Saved',run,metrics['character_accuracy'],metrics['digit_recall'],metrics['letter_false_positive_rate'],flush=True)


if __name__=='__main__':main()
