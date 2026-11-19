"""Apply grouping to a computed joint-MUSIC sequence without changing spectra."""
import json,csv
import numpy as np
from peak_grouping import GroupedTracker,GROUP_SETTINGS

def apply_grouping(out,maps,states,summary,expected_people=1):
    rawstates=[dict(s) for s in states];g=GroupedTracker(expected_people);new=[]
    for k,s in enumerate(rawstates):
        result=g.step(s['candidates'],maps['joint'][k],maps['range_power'][k],maps['angles'],k)
        q=dict(s);q['reflection_visible']=s['visible'];q['reflection_lost']=s['lost'];q.update(result);q['visible']=result['person_candidates'];q['lost']=result['lost_group_tracks'];new.append(q)
    tracks=[t for t in g.all() if t['confirmed']];summary=dict(summary)
    summary['reflection_tracks']=summary['tracks'];summary['tracks']=[dict(id=t['id'],history=t['history']) for t in tracks]
    summary['grouping']=dict(settings=GROUP_SETTINGS,expected_people=expected_people,raw_peak_count=sum(len(s['candidates']) for s in new),group_count=sum(len(s['groups']) for s in new),multi_peak_groups=sum(sum(q['member_count']>1 for q in s['groups']) for s in new),ambiguous_group_count=sum(len(s['ambiguous_groups']) for s in new),displayed_person_windows=sum(bool(s['person_candidates']) for s in new),max_displayed_person_candidates=max(len(s['person_candidates']) for s in new),group_track_segments=len(tracks),primary_ids=sorted(set(s['primary_id'] for s in new if s['primary_id'] is not None)))
    (out/'reflection_states.json').write_text(json.dumps(rawstates,indent=2));(out/'frame_states.json').write_text(json.dumps(new,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'grouping_settings.json').write_text(json.dumps(dict(**GROUP_SETTINGS,expected_people=expected_people),indent=2))
    with (out/'tracks.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['frame','start_observation','id','range_bin','angle_deg','confirmed','support','member_count']);w.writeheader()
        for k,s in enumerate(new):
            for t in s['person_candidates']:w.writerow(dict(frame=k,start_observation=int(maps['starts'][k]),**t))
    return new,summary
