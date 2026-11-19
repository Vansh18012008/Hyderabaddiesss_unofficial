"""Apply grouping to a computed joint-MUSIC sequence without changing spectra."""
import json,csv
import numpy as np
from peak_grouping import GroupedTracker,GROUP_SETTINGS

def _peak_metrics(z,power,rb,angle,angles):
    ai=int(np.argmin(abs(angles-angle)));ri=int(np.clip(round(rb),8,len(power)-2));lo=max(8,ri-10);hi=min(len(power)-1,ri+11);alo=max(0,ai-4);ahi=min(len(angles),ai+5)
    patch=np.array(z[alo:ahi,lo:hi],copy=True);r0=ri-lo;a0=ai-alo;patch[max(0,a0-1):a0+2,max(0,r0-2):r0+3]=np.nan
    ring=float(np.nanmedian(patch)) if np.isfinite(patch).any() else float(z[ai,ri]);local=float(z[ai,ri]-ring);contrast=float(z.max()-np.median(z[:,8:]))
    p_db=10*np.log10(np.maximum(power,1e-30));excess=float(p_db[ri]-np.median(p_db[8:]))
    if contrast>=8 and z[ai,ri]>=-15 and excess>=8 and local>=3:confidence='high'
    elif contrast>=6 and z[ai,ri]>=-22 and excess>=4 and local>=1:confidence='medium'
    else:confidence='low'
    return dict(range_bin=float(rb),angle_deg=float(angle),confidence=confidence,map_contrast_db=contrast,peak_level_db=float(z[ai,ri]),local_prominence_db=local,range_excess_db=excess,range_edge=bool(ri<12 or ri>243))

def _candidate_for_frame(q,z,power,angles,min_presence_contrast_db=15.0):
    groups=q['groups'];selected=q['person_candidates'];group=None
    if selected:
        person=selected[0];group=min(groups,key=lambda g:abs(g['range_bin']-person['range_bin'])+abs(g['angle_deg']-person['angle_deg']))
        status='associated_group';ambiguous=False
    elif q['ambiguous_groups']:
        group=max(q['ambiguous_groups'],key=lambda g:g['score_db']);status='ambiguous_group';ambiguous=True
    elif groups:
        group=max(groups,key=lambda g:g['score_db']+2*np.log1p(g['member_count']));status='unconfirmed_group';ambiguous=bool(group['ambiguous'])
    if group is None:
        # Find the strongest in-grid MUSIC maximum as a candidate only if the map
        # has enough contrast over its own background.
        area=z[1:-1,8:-1];ai0,ri0=np.unravel_index(np.argmax(area),area.shape);ai=ai0+1;ri=ri0+8
        metrics=_peak_metrics(z,power,float(ri),float(angles[ai]),angles)
        metrics.update(source='spectrum_fallback',group_id=None,member_count=0,confidence='low',status='no_supported_peak',ambiguous=False,candidate_available=True)
    else:
        metrics=_peak_metrics(z,power,group['range_bin'],group['angle_deg'],angles)
        visible=q.get('visible_groups',[]);matched=min(visible,key=lambda t:abs(t['range_bin']-group['range_bin'])+abs(t['angle_deg']-group['angle_deg'])) if visible else None;track_id=selected[0]['id'] if selected else (matched['id'] if matched and abs(matched['range_bin']-group['range_bin'])<1e-6 and abs(matched['angle_deg']-group['angle_deg'])<1e-6 else None)
        metrics.update(source=status,group_id=track_id,member_count=int(group['member_count']),confidence='ambiguous' if ambiguous else metrics['confidence'],status='ambiguous' if ambiguous else status,ambiguous=ambiguous,candidate_available=True,possible_track_ids=group.get('possible_track_ids',[]))
    if metrics['map_contrast_db'] < min_presence_contrast_db:
        return dict(range_bin=None,angle_deg=None,confidence='none',status='below_presence_threshold',source='none',group_id=None,member_count=0,map_contrast_db=metrics['map_contrast_db'],peak_level_db=metrics['peak_level_db'],local_prominence_db=metrics['local_prominence_db'],range_excess_db=metrics['range_excess_db'],range_edge=metrics['range_edge'],ambiguous=False,candidate_available=False,possible_track_ids=[])
    return metrics

def apply_grouping(out,maps,states,summary,expected_people=1,min_presence_contrast_db=15.0):
    rawstates=[dict(s) for s in states];g=GroupedTracker(expected_people);new=[]
    for k,s in enumerate(rawstates):
        result=g.step(s['candidates'],maps['joint'][k],maps['range_power'][k],maps['angles'],k)
        q=dict(s);q['reflection_visible']=s['visible'];q['reflection_lost']=s['lost'];q.update(result);q['candidate_location']=_candidate_for_frame(q,maps['joint'][k],maps['range_power'][k],maps['angles'],min_presence_contrast_db);q['visible']=result['person_candidates'];q['lost']=result['lost_group_tracks'];new.append(q)
    tracks=[t for t in g.all() if t['confirmed']];summary=dict(summary)
    summary['reflection_tracks']=summary['tracks'];summary['tracks']=[dict(id=t['id'],history=t['history']) for t in tracks]
    summary['grouping']=dict(settings=GROUP_SETTINGS,expected_people=expected_people,raw_peak_count=sum(len(s['candidates']) for s in new),group_count=sum(len(s['groups']) for s in new),multi_peak_groups=sum(sum(q['member_count']>1 for q in s['groups']) for s in new),ambiguous_group_count=sum(len(s['ambiguous_groups']) for s in new),displayed_person_windows=sum(bool(s['person_candidates']) for s in new),max_displayed_person_candidates=max(len(s['person_candidates']) for s in new),group_track_segments=len(tracks),primary_ids=sorted(set(s['primary_id'] for s in new if s['primary_id'] is not None)))
    summary['grouping'].update(min_presence_contrast_db=min_presence_contrast_db,candidate_locations_all_frames=sum(q['candidate_location']['candidate_available'] for q in new),frames_below_presence_threshold=sum(not q['candidate_location']['candidate_available'] for q in new),confidence_counts={v:sum(q['candidate_location']['confidence']==v for q in new) for v in ['high','medium','low','ambiguous','none']},fallback_spectrum_frames=sum(q['candidate_location']['source']=='spectrum_fallback' for q in new),frames_without_confirmed_person_track=sum(not q['person_candidates'] for q in new))
    (out/'reflection_states.json').write_text(json.dumps(rawstates,indent=2));(out/'frame_states.json').write_text(json.dumps(new,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'grouping_settings.json').write_text(json.dumps(dict(**GROUP_SETTINGS,expected_people=expected_people),indent=2))
    with (out/'tracks.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['frame','start_observation','id','range_bin','angle_deg','confirmed','support','member_count']);w.writeheader()
        for k,s in enumerate(new):
            for t in s['person_candidates']:w.writerow(dict(frame=k,start_observation=int(maps['starts'][k]),**t))
    with (out/'candidate_locations.csv').open('w') as f:
        fields=['frame','start_observation','candidate_available','range_bin','angle_deg','confidence','status','source','group_id','member_count','map_contrast_db','peak_level_db','local_prominence_db','range_excess_db','range_edge','ambiguous','possible_track_ids']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for k,q in enumerate(new):w.writerow(dict(frame=k,start_observation=int(maps['starts'][k]),**q['candidate_location']))
    return new,summary
