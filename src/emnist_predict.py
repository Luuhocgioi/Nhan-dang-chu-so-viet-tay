"""Đề xuất vùng nét và phân loại 62 lớp; chỉ xuất các lớp số đủ điểm."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .config import ROOT
from .emnist_model import CharacterCNN,CHARACTERS,normalize_glyph
from .preprocess import EmptyImageError


def merge_small_fragments(labels,stats):
    """Gộp nét đứt theo chồng lấp ngang/gap dọc nhỏ; không thêm pixel nét."""
    stats=stats.copy();labels=labels.copy()
    valid=[i for i in range(1,len(stats)) if stats[i,4]>=8]
    if len(valid)<50:return labels,stats
    height=float(np.median(stats[valid,3]));gap_limit=max(2,.2*height)
    for i in valid:
        if stats[i,4]==0:continue
        x,y,w,h,area=map(int,stats[i])
        for j in valid:
            if j<=i or stats[j,4]==0:continue
            xx,yy,ww,hh,aa=map(int,stats[j])
            overlap=min(x+w,xx+ww)-max(x,xx)
            gap=max(y,yy)-min(y+h,yy+hh)
            if min(h,hh)>.7*height or overlap<.3*min(w,ww) or gap<0 or gap>gap_limit:continue
            top=min(y,yy);bottom=max(y+h,yy+hh)
            if bottom-top>1.5*height:continue
            left=min(x,xx);right=max(x+w,xx+ww)
            labels[labels==j]=i;stats[j]=0
            stats[i]=[left,top,right-left,bottom-top,area+aa]
            x,y,w,h,area=map(int,stats[i])
    return labels,stats


def tiled_foreground(gray):
    """Ảnh ghép ô sáng trên nền tối: tách từng ô, bỏ viền, lấy nét bên trong.

    Chỉ kích hoạt khi có >=3 ô chữ nhật tương tự kích thước và viền ảnh tối.
    Không giả định số hàng/cột hoặc nhãn ký tự.
    """
    border=np.concatenate((gray[0],gray[-1],gray[:,0],gray[:,-1]))
    if np.median(border)>40 or np.mean(gray<40)<.35:return None
    _,bright=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,_=cv2.findContours(bright,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cells=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if min(w,h)<24 or not .6<w/h<1.6:continue
        if cv2.contourArea(contour)/(w*h)<.85:continue
        if w>gray.shape[1]*.5 or h>gray.shape[0]*.7:continue
        cells.append((x,y,w,h))
    if len(cells)<3:return None
    size=np.median([w*h for x,y,w,h in cells])
    cells=[b for b in cells if .5*size<=b[2]*b[3]<=2*size]
    if len(cells)<3:return None
    mask=np.zeros_like(gray)
    for x,y,w,h in cells:
        margin=max(2,round(min(w,h)*.04))
        inner=gray[y+margin:y+h-margin,x+margin:x+w-margin]
        if int(inner.max())-int(inner.min())<15:continue
        _,ink=cv2.threshold(inner,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        mask[y+margin:y+h-margin,x+margin:x+w-margin]=ink
    return mask


def prepare_page(image, mode='auto'):
    from .detect import to_bgr
    base=to_bgr(image)
    if max(base.shape[:2])>1800:
        scale=1800/max(base.shape[:2]);base=cv2.resize(base,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    if mode=='auto':
        tile_mask=tiled_foreground(cv2.cvtColor(base,cv2.COLOR_BGR2GRAY))
        if tile_mask is not None:return base,tile_mask,'tiles'
    hsv=cv2.cvtColor(base,cv2.COLOR_BGR2HSV)
    hue=(hsv[:,:,0]>=90)&(hsv[:,:,0]<=175)&(hsv[:,:,2]<=250)
    saturations=hsv[:,:,1][hue]
    saturation_threshold=20
    if len(saturations):
        split,_=cv2.threshold(saturations.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        saturation_threshold=max(20,min(60,int(split*.8)))
    color=cv2.inRange(hsv,np.array([90,saturation_threshold,0]),np.array([175,255,250]))
    red=cv2.bitwise_or(cv2.inRange(hsv,np.array([0,30,0]),np.array([15,255,250])),
                      cv2.inRange(hsv,np.array([170,30,0]),np.array([179,255,250])))
    def stroke_area(candidate):
        joined=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        _,_,stats,_=cv2.connectedComponentsWithStats(joined)
        total=0
        for x,y,w,h,area in stats[1:]:
            if x==0 or y==0 or x+w>=candidate.shape[1] or y+h>=candidate.shape[0]:continue
            if area>=12 and h>=12 and w>=2 and w/h<=1.8 and h<min(candidate.shape)*.5:total+=int(area)
        return total
    red_selected=mode=='auto' and stroke_area(red)>stroke_area(color)
    if red_selected:color=red
    use_color=mode=='color' or (mode=='auto' and np.count_nonzero(color)>max(40,base.shape[0]*base.shape[1]*.00005))
    if use_color:
        mask=color
        if red_selected:
            # Ảnh nén có viền ký tự đen: chỉ lấy pixel đỏ sẽ làm mất nét.
            # Dùng đỏ để định vị, nhưng giữ toàn bộ nét tối quanh vị trí đó.
            gray=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
            size=max(15,int(min(gray.shape)*.12)|1)
            background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
            ink=255-cv2.divide(gray,np.maximum(background,1),scale=255)
            values=ink[red>0]
            if len(values):
                split,_=cv2.threshold(values.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                _,dark=cv2.threshold(ink,max(18,split*.85),255,cv2.THRESH_BINARY)
                n,components,_,_=cv2.connectedComponentsWithStats(dark)
                seed=cv2.dilate(red,np.ones((3,3),np.uint8))>0
                keep=np.unique(components[seed]);keep=keep[keep!=0]
                mask=np.where(np.isin(components,keep),dark,0).astype(np.uint8)
    else:
        gray=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
        size=max(15,int(min(gray.shape)*.06)|1)
        background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
        ink=255-cv2.divide(gray,np.maximum(background,1),scale=255)
        threshold,_=cv2.threshold(ink,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _,mask=cv2.threshold(ink,max(18,threshold),255,cv2.THRESH_BINARY)
    # Nét chữ ở ảnh nhỏ chỉ rộng 1–2px; closing 3px có thể lấp ruột 8/9/4.
    if min(mask.shape)>=400:
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return base,mask,'color' if use_color else 'gray'


class EMNISTPredictor:
    display_name='EMNIST 62 lớp'

    def __init__(self, model_path=None, confidence=.65, mode='auto', numeric_rows=False):
        if model_path is None:
            pointer=ROOT/'models/emnist_candidate.json'
            if not pointer.exists():raise FileNotFoundError('Chưa có mô hình EMNIST. Chạy python -m src.train_emnist.')
            model_path=ROOT/json.loads(pointer.read_text(encoding='utf-8'))['model']
        self.path=Path(model_path);self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=CharacterCNN().to(self.device)
        self.model.load_state_dict(torch.load(self.path,map_location=self.device,weights_only=True));self.model.eval()
        self.confidence=confidence;self.mode=mode;self.numeric_rows=numeric_rows
        self.display_name='Dãy số rời — EMNIST + CNN' if numeric_rows else 'EMNIST 62 lớp'
        self.digit_classifier=None

    def recognize(self,image):
        base,mask,mode=prepare_page(image,self.mode)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        if mode=='gray':labels,stats=merge_small_fragments(labels,stats)
        proposals=[];glyphs=[]
        for i in range(1,count):
            x,y,w,h,area=map(int,stats[i])
            if x==0 or y==0 or x+w>=mask.shape[1] or y+h>=mask.shape[0]:continue
            if area<12 or h<12 or w<2 or w/h>1.8 or h>min(mask.shape)*.5:continue
            component=np.where(labels[y:y+h,x:x+w]==i,mask[y:y+h,x:x+w],0).astype(np.uint8)
            proposals.append([x,y,w,h]);glyphs.append(normalize_glyph(component))
        if not proposals:raise EmptyImageError('Không tìm thấy vùng nét phù hợp. Thử chọn vùng giấy hoặc kiểm tra màu mực.')
        with torch.inference_mode():
            x=torch.from_numpy(np.stack(glyphs)[:,None]).float().to(self.device)/255
            probabilities=torch.cat([self.model(chunk).softmax(1) for chunk in x.split(256)]).cpu().numpy()
        numeric_indices=set();digit_scores={}
        if self.numeric_rows:
            # Chỉ áp dụng giả định hàng số cho các ký tự đủ lớn, cùng hàng,
            # tách rời rộng hơn chữ trong từ. Không suy ra số từ vị trí cố định.
            large_height=float(np.percentile([b[3] for b in proposals],75))
            groups=[]
            for index,b in enumerate(proposals):
                x,y,w,h=b
                if h<.6*large_height:continue
                for group in groups:
                    centers=[proposals[j][1]+proposals[j][3]/2 for j in group]
                    if abs(y+h/2-np.median(centers))<.5*h:group.append(index);break
                else:groups.append([index])
            for group in groups:
                group.sort(key=lambda j:proposals[j][0])
                heights=np.array([proposals[j][3] for j in group])
                centers=np.array([proposals[j][0]+proposals[j][2]/2 for j in group])
                if len(group)>=3 and heights.max()/heights.min()<1.8 and np.median(np.diff(centers))>1.8*np.median(heights):
                    numeric_indices.update(group)
            # Lưới dày cần khoảng cách thấp hơn dãy rời, nhưng phải có nhiều
            # hàng và nhịp cột đều. Không dùng vị trí hàng để gán nhãn.
            dense_groups=[]
            for group in groups:
                if len(group)<8:continue
                ordered=sorted(group,key=lambda j:proposals[j][0])
                centers=np.array([proposals[j][0]+proposals[j][2]/2 for j in ordered])
                gaps=np.diff(centers);heights=np.array([proposals[j][3] for j in ordered])
                if np.std(gaps)/max(1,np.mean(gaps))<.35 and np.median(gaps)>.7*np.median(heights):dense_groups.append(ordered)
            if len(dense_groups)>=3:
                for group in dense_groups:numeric_indices.update(group)
            if numeric_indices:
                if self.digit_classifier is None:
                    from .predict import Predictor
                    self.digit_classifier=Predictor()
                from .preprocess import preprocess
                inputs=[];indices=sorted(numeric_indices)
                for index in indices:
                    x,y,w,h=proposals[index]
                    component=mask[y:y+h,x:x+w]
                    inputs.append(preprocess(cv2.copyMakeBorder(component,5,5,5,5,cv2.BORDER_CONSTANT,value=0),polarity='light'))
                scores=self.digit_classifier.model(np.concatenate(inputs),training=False).numpy()
                digit_scores=dict(zip(indices,scores))
        rows=[];rejected=[]
        for index,(bbox,glyph,p) in enumerate(zip(proposals,glyphs,probabilities)):
            label=int(p.argmax());score=float(p[label])
            if index in digit_scores:
                p=digit_scores[index];label=int(p.argmax());score=float(p[label])
            elif self.numeric_rows and numeric_indices:
                rejected.append({'bbox':bbox,'character':CHARACTERS[label],'score':score,'reason':'outside separated numeric rows'});continue
            if label>=10 or score<self.confidence:
                rejected.append({'bbox':bbox,'character':CHARACTERS[label],'score':score});continue
            detection={'digit':label,'score':score,'bbox':bbox,'probabilities':p[:10].tolist(),'glyph':glyph}
            x,y,w,h=bbox
            for row in rows:
                top=min(d['bbox'][1] for d in row);bottom=max(d['bbox'][1]+d['bbox'][3] for d in row)
                if min(y+h,bottom)-max(y,top)>=.4*min(h,bottom-top):row.append(detection);break
            else:rows.append([detection])
        annotated=base.copy();detections=[];thumbs=[];lines=[]
        for row_id,row in enumerate(sorted(rows,key=lambda r:min(d['bbox'][1] for d in r))):
            row.sort(key=lambda d:d['bbox'][0]);lines.append(''.join(str(d['digit']) for d in row))
            for d in row:
                d['row']=row_id;thumbs.append(d.pop('glyph'));detections.append(d)
                x,y,w,h=d['bbox'];cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,255,0),2)
                font_scale=max(.3,min(.9,h/50))
                label_height=cv2.getTextSize(str(d['digit']),cv2.FONT_HERSHEY_SIMPLEX,font_scale,1)[0][1]
                cv2.putText(annotated,str(d['digit']),(x,max(label_height+1,y-3)),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(0,255,255),1 if h<35 else 2,cv2.LINE_AA)
        return {'text':'\n'.join(lines),'detections':detections,'rejected':rejected,'backend':'numeric-rows' if self.numeric_rows else 'emnist62',
                'confidence_threshold':self.confidence,'proposal_mode':mode,'coordinate_space':'output_image',
                'assumption':'Separated numeric rows or >=3 regular dense rows with >=8 symbols each; letter tables may still be mistaken.' if self.numeric_rows else 'No numeric-row assumption',
                'input_size':[image.shape[1],image.shape[0]],'output_size':[base.shape[1],base.shape[0]],
                'message':'Không có chữ số đủ điểm nhận dạng.' if not detections else ''},annotated,thumbs
