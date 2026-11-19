"""Rolling exploratory BR/HR estimates from the grouped range-angle candidate."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,welch

def _peak_rate(x,fs,band,min_seconds=20.):
    x=np.asarray(x,float);x=x-np.mean(x)
    if len(x)<int(min_seconds*fs):return None,0.
    sos=butter(3,band,btype='bandpass',fs=fs,output='sos');y=sosfiltfilt(sos,x)
    nfft=max(8192,2**int(np.ceil(np.log2(len(y)*8))));f,p=welch(y,fs=fs,nperseg=len(y),nfft=nfft,window='hann')
    keep=(f>=band[0])&(f<=band[1]);
    if np.count_nonzero(keep)<3:return None,0.
    fb,pb=f[keep],p[keep];i=int(np.argmax(pb));floor=float(np.median(pb));prom=float(10*np.log10(max(pb[i],1e-30)/max(floor,1e-30)))
    return float(fb[i]*60),prom

def analyze_vitals(cube,maps,states,cfg,outdir):
    fs=float(cfg.get('slow_fs_hz') or 20.);nobs=cube.shape[1];nfft=cube.shape[2];nrx=cube.shape[0];window=int(maps['window_observations']);starts=np.asarray(maps['starts']);centers=starts+(window-1)/2
    loc=[s['candidate_location'] for s in states];available=np.array([bool(q.get('candidate_available',True)) for q in loc]);bins=np.array([q['range_bin'] if q.get('range_bin') is not None else 0. for q in loc],float);angs=np.array([q['angle_deg'] if q.get('angle_deg') is not None else 0. for q in loc],float);conf=np.array([q.get('confidence','none') for q in loc]);gid=np.array([q.get('group_id') if q.get('group_id') is not None else -1 for q in loc]);src=np.array([q.get('source','') for q in loc])
    # Assign each slow-time observation the nearest measured frame candidate.
    frame_idx=np.clip(np.searchsorted(centers,np.arange(nobs)),0,len(centers)-1);left=np.maximum(frame_idx-1,0);choose_left=abs(np.arange(nobs)-centers[left])<abs(np.arange(nobs)-centers[frame_idx]);frame_idx=np.where(choose_left,left,frame_idx)
    rb=bins[frame_idx];ang=angs[frame_idx];
    rp=np.fft.fft(cube*np.hanning(nfft)[None,None,:],axis=-1)[:,:,:nfft//2]
    rx=np.arange(nrx);wl=3e8/float(cfg['carrier_hz']);sp=cfg['antenna_spacing_m']/wl;w=np.exp(-1j*cfg['angular_sign']*2j*np.pi*sp*rx[:,None]*np.sin(np.deg2rad(ang))[None,:])/nrx
    rb=np.clip(rb,1,nfft//2-2);r0=np.floor(rb).astype(int);frac=rb-r0;cols=np.arange(nobs);a=rp[:,cols,r0];b=rp[:,cols,r0+1];interp=a*(1-frac)[None,:]+b*frac[None,:];iq=np.sum(w*interp,axis=0)
    # A frame is trusted only when its candidate is tied to an actual group and is at least medium support.
    trusted=available[frame_idx]&np.isin(conf[frame_idx],['high','medium'])&(gid[frame_idx]>=0)&np.isin(src[frame_idx],['associated_group','unconfirmed_group']);amp=np.abs(iq);phase=np.unwrap(np.angle(iq));duration=20.;need=int(np.ceil(duration*fs));rows=[];sig=[]
    for fi,center in enumerate(centers):
        end=int(round(center));lo=end-need+1
        row=dict(frame=fi,start_observation=int(starts[fi]),end_observation=int(end),slow_fs_assumed_hz=fs,window_seconds=duration,br_bpm=None,hr_bpm=None,exploratory_br_bpm=None,exploratory_hr_bpm=None,exploratory_window_seconds=None,exploratory_label='unvalidated',diagnostic_br_peak_bpm=None,diagnostic_hr_peak_bpm=None,diagnostic_br_prominence_db=None,diagnostic_hr_prominence_db=None,resp_peak_prominence_db=None,heart_peak_prominence_db=None,group_coverage=0.,unique_group_ids=0,range_span_bins=None,angle_span_deg=None,status='warming_up')
        # Less restrictive display-only estimate: use up to 20 s of available
        # phase, beginning after 10 s. It deliberately ignores tracking gates.
        if available[fi] and 0<=end<nobs and end+1>=int(10*fs):
            exp_n=min(need,end+1);exp_q=slice(end-exp_n+1,end+1);exp_disp=phase[exp_q]*wl/(4*np.pi)*1000
            exp_br,_=_peak_rate(exp_disp,fs,(.1,.5),min_seconds=10.);exp_hr,_=_peak_rate(exp_disp,fs,(.8,2.5),min_seconds=10.)
            row.update(exploratory_br_bpm=exp_br,exploratory_hr_bpm=exp_hr,exploratory_window_seconds=exp_n/fs)
        if lo<0 or end>=nobs:
            row['status']='insufficient_history';rows.append(row);sig.append(dict(phase=np.nan,iq_amp=np.nan));continue
        q=slice(lo,end+1);disp=phase[q]*wl/(4*np.pi)*1000;disp=disp-np.mean(disp);br_diag,bp_diag=_peak_rate(disp,fs,(.1,.5));hr_diag,hp_diag=_peak_rate(disp,fs,(.8,2.5));row.update(diagnostic_br_peak_bpm=br_diag,diagnostic_hr_peak_bpm=hr_diag,diagnostic_br_prominence_db=bp_diag,diagnostic_hr_prominence_db=hp_diag);ok=trusted[q];coverage=float(np.mean(ok));uids=np.unique(gid[frame_idx[q]][ok]);rbw=rb[q][ok];aw=ang[q][ok]
        row.update(group_coverage=coverage,unique_group_ids=int(len(uids)),range_span_bins=float(np.ptp(rbw)) if len(rbw) else None,angle_span_deg=float(np.ptp(aw)) if len(aw) else None)
        # Guard against averaging a person walking through different range/angle cells.
        if coverage<.8:row['status']='insufficient_tracked_signal'
        elif len(uids)!=1:row['status']='track_switch_or_ambiguity'
        elif np.ptp(rbw)>6 or np.ptp(aw)>6:row['status']='walking_motion_uncompensated'
        elif np.nanpercentile(amp[q],10)<=0 or np.nanpercentile(amp[q],90)/max(np.nanpercentile(amp[q],10),1e-12)>30:row['status']='unstable_radar_amplitude'
        else:
            # Phase-only estimates use the established pipeline's bands and FFT peak method.
            br,bp=br_diag,bp_diag;hr,hp=hr_diag,hp_diag;row['resp_peak_prominence_db']=bp;row['heart_peak_prominence_db']=hp
            if br is None or hr is None:row['status']='no_in_band_peak'
            elif bp<4 or hp<4:row['status']='weak_rate_peak'
            else:row.update(br_bpm=br,hr_bpm=hr,status='provisional_estimate')
        rows.append(row);sig.append(dict(phase=float(phase[end]),iq_amp=float(amp[end])))
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/'vital_estimates.csv').open('w') as f:
        wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
    np.savez_compressed(outdir/'vital_phase.npz',iq=iq,phase_rad=phase,displacement_mm=phase*wl/(4*np.pi)*1000,slow_fs_assumed_hz=fs,window_index=frame_idx,trusted_sample=trusted,range_bin=rb,angle_deg=ang)
    def diagnostic_stats(key):
        vals=[r[key] for r in rows if r[key] is not None]
        return dict(count=len(vals),median=float(np.median(vals)) if vals else None,min=float(np.min(vals)) if vals else None,max=float(np.max(vals)) if vals else None)
    summary=dict(slow_rate_hz=fs,slow_rate_verified=cfg.get('slow_fs_hz') is not None,window_seconds=duration,window_samples=need,frames=len(rows),frames_with_reportable_rates=sum(r['status']=='provisional_estimate' for r in rows),frames_with_exploratory_rates=sum(r['exploratory_br_bpm'] is not None and r['exploratory_hr_bpm'] is not None for r in rows),exploratory_br_range=[r['exploratory_br_bpm'] for r in rows if r['exploratory_br_bpm'] is not None],exploratory_hr_range=[r['exploratory_hr_bpm'] for r in rows if r['exploratory_hr_bpm'] is not None],statuses={x:sum(r['status']==x for r in rows) for x in sorted(set(r['status'] for r in rows))},br_range=[r['br_bpm'] for r in rows if r['br_bpm'] is not None],hr_range=[r['hr_bpm'] for r in rows if r['hr_bpm'] is not None],diagnostic_br_fft_peaks=diagnostic_stats('diagnostic_br_peak_bpm'),diagnostic_hr_fft_peaks=diagnostic_stats('diagnostic_hr_peak_bpm'),limitations=['Exploratory rates use up to 20 s of candidate phase and may start at 10 s; they ignore the reliability gates and are not validated physiological measurements.','Rates use an assumed slow-time sampling rate when metadata is absent.','Diagnostic FFT peaks are computed even in rejected windows and must not be interpreted as BR/HR. Walking micro-motion is not compensated.','Any estimate needs synchronized physiological ground truth for validation.'])
    (outdir/'vital_summary.json').write_text(json.dumps(summary,indent=2));summary_file=outdir/'summary.json'
    if summary_file.exists():
        full=json.loads(summary_file.read_text());full['vitals']=summary;summary_file.write_text(json.dumps(full,indent=2))
    
    for r in rows:states[r['frame']]['vitals']=r
    (outdir/'frame_states_with_vitals.json').write_text(json.dumps(states,indent=2));(outdir/'frame_states.json').write_text(json.dumps(states,indent=2))
    return rows,summary
