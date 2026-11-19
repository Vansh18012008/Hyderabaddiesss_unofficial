"""Ungated-map candidate detection and two-threshold association.
Confidence labels describe threshold support, not human identity or probabilities.
"""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

SETTINGS=dict(strong_map_db=-15.,strong_range_db=8.,strong_contrast_db=8.,weak_map_db=-22.,weak_range_db=4.,weak_contrast_db=6.,min_bin=8,range_gate=16.,angle_gate=15.,confirm_hits=3,max_missing=2,max_weak_run=5,max_births_per_window=8,max_candidates=24)

def candidates(z,power,angles):
    s=SETTINGS;pdb=10*np.log10(np.maximum(power,1e-30));floor=np.median(pdb[s['min_bin']:]);contrast=float(z[:,s['min_bin']:].max()-np.median(z[:,s['min_bin']:]))
    if contrast<s['weak_contrast_db']:return []
    mask=(z==maximum_filter(z,size=(3,7)))&(z>=s['weak_map_db']);out=[]
    for ai,ri in sorted(np.argwhere(mask),key=lambda ij:z[tuple(ij)],reverse=True):
        excess=float(pdb[ri]-floor)
        if ri<s['min_bin'] or ri>=len(power)-1 or ai in (0,len(angles)-1) or excess<s['weak_range_db']:continue
        if any(abs(ri-q['range_bin'])<=5 and abs(angles[ai]-q['angle_deg'])<=10 for q in out):continue
        strong=bool(z[ai,ri]>=s['strong_map_db'] and excess>=s['strong_range_db'] and contrast>=s['strong_contrast_db'])
        out.append(dict(range_bin=int(ri),angle_deg=float(angles[ai]),score_db=float(z[ai,ri]),range_excess_db=excess,strong=strong))
        if len(out)>=s['max_candidates']:break
    return out

class ConfidenceTracker:
    def __init__(self,allow_turns=False):self.active=[];self.archive=[];self.next_id=1;self.allow_turns=allow_turns
    def step(self,dets,k):
        s=SETTINGS;used=set();matched=set()
        # Strong detections have priority. Weak detections cannot start or confirm tracks.
        for strong_stage in (True,False):
            ids=[i for i,t in enumerate(self.active) if i not in matched and (strong_stage or (t['confirmed'] and t['weak_run']<s['max_weak_run']))]
            js=[j for j,d in enumerate(dets) if j not in used and d['strong']==strong_stage]
            if not ids or not js:continue
            cost=np.full((len(ids),len(js)),1e6)
            for ii,i in enumerate(ids):
                t=self.active[i];dt=k-t['last'];pred=t['pos']+t['vel']*dt
                for jj,j in enumerate(js):
                    if dets[j].get('owner') is not None and dets[j]['owner']!=t['id']:continue
                    pos=np.array([dets[j]['range_bin'],dets[j]['angle_deg']]);delta=abs(pos-pred);travel=abs(pos-t['pos'])
                    within_travel=travel[0]<=s['range_gate']*dt and travel[1]<=s['angle_gate']*dt
                    predicted_ok=delta[0]<=s['range_gate'] and delta[1]<=s['angle_gate']
                    if within_travel and predicted_ok:cost[ii,jj]=np.linalg.norm(delta/[s['range_gate'],s['angle_gate']])
                    elif within_travel and self.allow_turns and strong_stage:cost[ii,jj]=1.+np.linalg.norm(travel/[s['range_gate']*dt,s['angle_gate']*dt])
            ii,jj=linear_sum_assignment(cost)
            for a,b in zip(ii,jj):
                if cost[a,b]>=1e5:continue
                i,j=ids[a],js[b];t=self.active[i];d=dets[j];pos=np.array([d['range_bin'],d['angle_deg']],float);dt=k-t['last'];t['vel']=np.clip(.5*t['vel']+.5*(pos-t['pos'])/dt,[-s['range_gate'],-s['angle_gate']],[s['range_gate'],s['angle_gate']]);t['pos']=pos;t['last']=k;t['missing']=0
                t['streak']=t['streak']+1 if strong_stage else 0;t['confirmed']=t['confirmed'] or t['streak']>=s['confirm_hits'];t['weak_run']=0 if strong_stage else t['weak_run']+1;t['support']='strong' if strong_stage else 'weak';t['history'].append([k,float(pos[0]),float(pos[1]),t['support']]);used.add(j);matched.add(i)
        keep=[];visible=[];lost=[]
        for i,t in enumerate(self.active):
            if i not in matched:t['missing']+=1;t['streak']=0
            if t['missing']:
                if t['confirmed']:lost.append(t['id'])
            else:visible.append(t)
            if t['missing']>s['max_missing']:self.archive.append(t)
            else:keep.append(t)
        self.active=keep;births=0
        for j,d in enumerate(dets):
            if j in used or not d['strong'] or d.get('owner') is not None or births>=s['max_births_per_window']:continue
            pos=np.array([d['range_bin'],d['angle_deg']],float);t=dict(id=self.next_id,pos=pos,vel=np.zeros(2),last=k,missing=0,streak=1,confirmed=False,weak_run=0,support='strong',history=[[k,float(pos[0]),float(pos[1]),'strong']]);self.next_id+=1;self.active.append(t);visible.append(t);births+=1
        return [dict(id=t['id'],range_bin=float(t['pos'][0]),angle_deg=float(t['pos'][1]),confirmed=t['confirmed'],support=t['support']) for t in visible],lost
    def all(self):return self.archive+self.active

def test_tracker():
    def d(r,strong=True):return [dict(range_bin=r,angle_deg=0,strong=strong)]
    t=ConfidenceTracker();v,_=t.step(d(50,False),0);assert not v and not t.active
    for k in range(3):v,_=t.step(d(50+2*k),k+1)
    assert v[0]['confirmed'] and v[0]['id']==1
    v,_=t.step(d(56,False),4);assert v[0]['id']==1 and v[0]['support']=='weak'
    v,lost=t.step(d(200,False),5);assert not v and 1 in lost
    t.step([],6);t.step([],7);assert not t.active
    t=ConfidenceTracker()
    for k in range(3):t.step(d(50),k)
    for k in range(3,8):v,_=t.step(d(50,False),k);assert v
    v,lost=t.step(d(50,False),8);assert not v and 1 in lost
    t=ConfidenceTracker();t.step(d(50),0);v,_=t.step(d(51,False),1);assert not v and not t.active[0]['confirmed']
    return 'PASS: weak candidates cannot start/confirm tracks; confirmed tracks accept nearby weak measurements; distant jumps rejected; missing tracks retired; weak continuation capped.'
