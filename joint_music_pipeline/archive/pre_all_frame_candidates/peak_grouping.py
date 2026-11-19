"""Provisional range-bin/angle grouping; no calibrated person classifier."""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment
from walking_confidence_tracker import ConfidenceTracker

GROUP_SETTINGS=dict(range_radius_bins=18.,angle_radius_deg=12.,valley_db=4.,deep_valley_db=8.,valley_persistence=2,motion_range_delta=4.,motion_angle_delta=4.,history_min=3,primary_min_range_span=8.)

class PeakGrouper:
    def __init__(self):self.previous=[];self.next_id=1;self.valleys={};self.audit=[]
    def group(self,dets,z,power,angles,k,tracks):
        s=GROUP_SETTINGS;peaks=[dict(d) for d in dets];used=set();assigned={}
        prev=[p for p in self.previous if k-p['last']<=2]
        if prev and peaks:
            cost=np.full((len(prev),len(peaks)),1e6)
            for i,p in enumerate(prev):
                for j,d in enumerate(peaks):
                    delta=np.array([d['range_bin'],d['angle_deg']])-p['pos'];dt=k-p['last']
                    if abs(delta[0])<=16*dt and abs(delta[1])<=15*dt:cost[i,j]=np.linalg.norm(delta/[16*dt,15*dt])
            ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj):
                if cost[i,j]>=1e5:continue
                p=prev[i];d=peaks[j];pos=np.array([d['range_bin'],d['angle_deg']],float);v=(pos-p['pos'])/(k-p['last']);assigned[j]=dict(id=p['id'],pos=pos,vel=.5*p['vel']+.5*v,age=p['age']+1,last=k);used.add(i)
        now=[]
        established=[t for t in tracks if t['confirmed'] and k-t['last']<=3]
        for j,d in enumerate(peaks):
            if j not in assigned:
                assigned[j]=dict(id=self.next_id,pos=np.array([d['range_bin'],d['angle_deg']],float),vel=np.zeros(2),age=1,last=k);self.next_id+=1
            p=assigned[j];now.append(p);d['_history']=p;owners=[]
            for t in established:
                pred=t['pos']+t['vel']*(k-t['last']);delta=abs(p['pos']-pred)
                if delta[0]<=18 and delta[1]<=12:owners.append((float(np.linalg.norm(delta/[18,12])),t['id']))
            owners.sort();d['_owners']=[q[1] for q in owners];d['_owner']=owners[0][1] if owners else None
        self.previous=now+[p for i,p in enumerate(prev) if i not in used and k-p['last']<=1]
        self.audit=[]
        def compatible(i,j):
            a,b=peaks[i],peaks[j];delta=abs(a['_history']['pos']-b['_history']['pos'])
            if delta[0]>s['range_radius_bins'] or delta[1]>s['angle_radius_deg']:return False
            reason=None
            if a['_owner'] is not None and b['_owner'] is not None and a['_owner']!=b['_owner']:reason='different_established_tracks'
            h1,h2=a['_history'],b['_history'];dv=abs(h1['vel']-h2['vel'])
            if min(h1['age'],h2['age'])>=s['history_min'] and (dv[0]>s['motion_range_delta'] or dv[1]>s['motion_angle_delta']):reason=reason or 'different_motion'
            ai=np.interp(a['angle_deg'],angles,np.arange(len(angles)));bi=np.interp(b['angle_deg'],angles,np.arange(len(angles)))
            rr=np.linspace(a['range_bin'],b['range_bin'],21);aa=np.linspace(ai,bi,21);profile=map_coordinates(z,[aa,rr],order=1,mode='nearest');dip=float(min(profile[0],profile[-1])-profile.min());key=tuple(sorted([h1['id'],h2['id']]))
            previous=self.valleys.get(key,(-2,0));count=(previous[1]+1 if previous[0]==k-1 else (previous[1] if previous[0]==k else 1)) if dip>=s['valley_db'] else 0;self.valleys[key]=(k,count)
            if dip>=s['deep_valley_db'] or count>=s['valley_persistence']:reason=reason or 'resolved_valley'
            self.audit.append(dict(peak_ids=list(key),range_separation=float(delta[0]),angle_separation=float(delta[1]),valley_db=dip,history_samples=min(h1['age'],h2['age']),merge=reason is None,reason=reason or 'close_connected_comoving_or_history_pending'))
            return reason is None
        groups=[]
        for i in range(len(peaks)):
            for g in groups:
                # Complete linkage prevents a chain of nearby peaks merging a wide scene.
                if all(compatible(i,j) for j in g):g.append(i);break
            else:groups.append([i])
        out=[]
        for g in groups:
            members=[peaks[j] for j in g];w=np.array([max(float(power[int(d['range_bin'])]),1e-30) for d in members]);w/=w.sum();rb=float(sum(v*d['range_bin'] for v,d in zip(w,members)));ang=float(sum(v*d['angle_deg'] for v,d in zip(w,members)))
            owners=set(d['_owner'] for d in members if d['_owner'] is not None);ambiguous=len(members)==1 and len(members[0]['_owners'])>1
            out.append(dict(range_bin=rb,angle_deg=ang,strong=any(d['strong'] for d in members),score_db=max(d['score_db'] for d in members),member_indices=g,member_count=len(g),owner=next(iter(owners)) if len(owners)==1 else None,ambiguous=ambiguous,possible_track_ids=members[0]['_owners'] if ambiguous else sorted(owners)))
        self.valleys={key:v for key,v in self.valleys.items() if k-v[0]<=3}
        return out

class GroupedTracker:
    def __init__(self,expected_people=1):self.grouper=PeakGrouper();self.tracker=ConfidenceTracker(allow_turns=True);self.primary=None;self.expected_people=expected_people
    def step(self,dets,z,power,angles,k):
        groups=self.grouper.group(dets,z,power,angles,k,self.tracker.active)
        usable=[g for g in groups if not g['ambiguous']];visible,lost=self.tracker.step(usable,k)
        for t in visible:
            g=min(usable,key=lambda q:abs(q['range_bin']-t['range_bin'])+abs(q['angle_deg']-t['angle_deg']));t['member_count']=g['member_count']
        active={t['id']:t for t in self.tracker.active};confirmed=[t for t in visible if t['confirmed']]
        if self.primary is not None and self.primary not in active:self.primary=None
        if self.primary is None:
            eligible=[]
            for t in confirmed:
                h=np.array([r[:3] for r in active[t['id']]['history']]);span=float(np.ptp(h[:,1]))
                if span>=GROUP_SETTINGS['primary_min_range_span']:eligible.append((len(h),min(span,50),t['id']))
            if eligible:self.primary=max(eligible)[2]
        selected=[t for t in confirmed if t['id']==self.primary] if self.expected_people==1 else confirmed
        status='observed' if selected else ('ambiguous_or_lost' if self.primary is not None else 'not_yet_acquired')
        return dict(groups=groups,visible_groups=visible,lost_group_tracks=lost,person_candidates=selected,primary_id=self.primary,status=status,ambiguous_groups=[g for g in groups if g['ambiguous']],pair_decisions=self.grouper.audit)
    def all(self):return self.tracker.all()
