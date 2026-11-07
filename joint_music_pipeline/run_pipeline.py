"""Portable joint range-angle MUSIC pipeline; output contains joint MUSIC only."""
from pathlib import Path
import argparse,json,hashlib,csv,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from group_output import apply_grouping
from adc_io import CPUIOEngine
from joint_music import steering_matrix,smoothed_covariance,spectrum_from_steering
from walking_confidence_tracker import candidates,ConfidenceTracker,SETTINGS,test_tracker
ROOT=Path(__file__).resolve().parent

def render(out,maps,states,summary,debug=False):
    if debug:
        states=[dict(q,visible=q["visible_groups"],lost=q["lost_group_tracks"]) for q in states]
    single=summary.get("grouping",{}).get("expected_people",1)==1
    frame_folder="debug_frames" if debug else "frames"
    gif_name="debug_groups.gif" if debug else "joint_music.gif"
    html_name="debug_replay.html" if debug else "replay.html"
    z=maps['joint'];angles=maps['angles'];bins=maps['range_bins'];starts=maps['starts'];count=len(starts);window=int(maps['window_observations']);(out/frame_folder).mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(13,5.5));fig.subplots_adjust(left=.065,right=.97,bottom=.2,top=.78,wspace=.25);files=[]
    for k,lo in enumerate(starts):
        for ax in axs:ax.clear()
        ax=axs[0];im=ax.imshow(z[k],origin='lower',aspect='auto',extent=[bins[0],bins[-1],angles[0],angles[-1]],vmin=-30,vmax=0,cmap='inferno');ax.set(title='Joint range–angle MUSIC | signal rank '+str(summary['signal_rank']),xlabel='Range / beat-frequency bin (nominal)',ylabel='Nominal angle (deg)')
        if debug:
            for d in states[k]['candidates']:ax.plot(d['range_bin'],d['angle_deg'],'x',color='white',ms=4,alpha=.65)
        for t in states[k]['visible']:
            col=('#27f5a0' if t['support']=='strong' else '#ffbc42') if t['confirmed'] else '#d4d4d4';ax.plot(t['range_bin'],t['angle_deg'],'o',mec=col,mfc='none',ms=11 if not debug else 8)
            if t['confirmed']:ax.annotate(f"{'G' if debug else 'P'}{t['id']}",(t['range_bin'],t['angle_deg']),xytext=(3,4),textcoords='offset points',fontsize=7,color=col)
        note=f"Peak above median: {states[k]['map_contrast_db']:.1f} dB"
        if not any(t['confirmed'] for t in states[k]['visible']):note+=' | person candidate uncertain/lost' if not debug else ' | no confirmed group'
        ax.text(.01,.98,note,transform=ax.transAxes,color='white',va='top',fontsize=8,bbox=dict(facecolor='black',alpha=.65,pad=2))
        lost_ids=states[k]['lost'] if debug else ([states[k]['primary_id']] if states[k]['primary_id'] in states[k]['lost'] else [])
        if lost_ids:ax.text(.01,.02,'Track lost: '+','.join(map(str,lost_ids)),transform=ax.transAxes,color='white',fontsize=8)
        ax=axs[1]
        for t in summary['tracks']:
            if not any(any(v['id']==t['id'] and v['confirmed'] for v in s['visible']) for s in states[:k+1]):continue
            curve=np.full(k+1,np.nan)
            for row in t['history']:
                if row[0]<=k:curve[row[0]]=row[1]
            ax.plot(starts[:k+1]+(window-1)/2,curve,'.-',lw=1,ms=3)
        ax.set(xlim=(0,summary['observations']),ylim=(bins[0],bins[-1]),title='Selected candidate-person history' if not debug else 'All group tracks: debugging',xlabel='Observation index',ylabel='Range / beat-frequency bin');ax.axvline(lo+(window-1)/2,color='gray',ls=':');ax.grid(alpha=.2)
        fig.suptitle(f"{summary['input_name']} | window {k+1}/{count} | observations {lo}–{lo+window-1}\nJoint MUSIC + four-evidence grouping | {'all reflections/groups (DEBUG)' if debug else ('single-person prior; one selected group' if single else 'all confirmed candidate groups; person count unverified')}",fontsize=13)
        if k==0:fig.text(.5,.045,('DEBUG: white crosses = raw reflections; circles = groups. Group IDs are not verified people.' if debug else ('One candidate-person circle uses the known single-person recording prior; identity remains unverified.' if single else 'Confirmed groups are candidate people, not verified identities or a validated person count.'))+'\nGreen: strong support | Amber: weaker support. Range-bin/angle grouping thresholds are provisional.',ha='center',fontsize=9)
        f=out/frame_folder/f'frame_{k:03d}.png';fig.savefig(f,dpi=100);files.append(f)
        if k%20==0:print(f'Rendered {k+1}/{count}',flush=True)
    plt.close(fig)
    imgs=[Image.open(f).convert('RGB').resize((1170,495)).quantize(colors=128) for f in files];imgs[0].save(out/gif_name,save_all=True,append_images=imgs[1:],duration=200,loop=0,optimize=False)
    images=['data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode() for f in files]
    html='''<!doctype html><meta charset="utf-8"><title>Joint MUSIC replay</title><style>body{margin:24px auto;max-width:1300px;font:16px system-ui;background:#f5f6f8;color:#172333}img{width:100%}button{padding:8px 16px}input{width:55%}</style><h1>Joint range–angle MUSIC</h1><p>Green: confirmed strong measurement. Amber: weaker continuation. Gray: tentative. Track IDs do not identify people.</p><button id="play">Play</button> <button id="prev">Previous</button> <button id="next">Next</button> <input id="scrub" type="range" min="0" max="MAX" value="0"> <span id="label"></span><img id="frame"><p>These are overlapping processing windows, not synchronized video frames. Playback speed is illustrative. A bright flat map does not establish a target. Range is shown in bins; physical timing and distance calibration remain unverified.</p><script>const frames=IMAGES;let i=0,timer=null;const slider=document.getElementById('scrub');function show(){document.getElementById('frame').src=frames[i];slider.value=i;document.getElementById('label').textContent=(i+1)+' / '+frames.length;}slider.oninput=()=>{i=+slider.value;show()};document.getElementById('next').onclick=()=>{i=(i+1)%frames.length;show()};document.getElementById('prev').onclick=()=>{i=(i+frames.length-1)%frames.length;show()};document.getElementById('play').onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent='Play'}else{timer=setInterval(()=>{i=(i+1)%frames.length;show()},200);this.textContent='Pause'}};show();</script>'''.replace('MAX',str(count-1)).replace('IMAGES',json.dumps(images))
    html=html.replace('<h1>Joint range–angle MUSIC</h1>', '<h1>Joint MUSIC: grouped candidate person</h1><p><a href="debug_replay.html">Show all reflection peaks and groups</a> | <a href="replay.html">Main single-person view</a></p>')
    html=html.replace('Track IDs do not identify people.', 'Grouping uses separation, spectral valleys, motion history and existing tracks. Main view uses the known single-person prior; identity is unverified.')
    (out/html_name).write_text(html)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,default=ROOT/'data/RATHAN_WALKING_VIDEO.bin');ap.add_argument('--config',type=Path,default=ROOT/'config.json');ap.add_argument('--output',type=Path,default=ROOT/'output');ap.add_argument('--render-only',action='store_true');args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        with np.load(out/'maps.npz') as a:maps={k:a[k] for k in a.files}
        states=json.loads((out/'frame_states.json').read_text());summary=json.loads((out/'summary.json').read_text());render(out,maps,states,summary);render(out,maps,states,summary,debug=True);return
    c=json.loads(args.config.read_text());n=c['n_samples'];nr=c['num_rx'];stride=nr*n*4;size=args.input.stat().st_size
    if size%stride:raise ValueError('BIN byte length is incompatible with the configured int16 complex IQ layout')
    observations=size//stride;window=c['window_observations'];hop=c['hop_observations']
    if observations<window:raise ValueError('Recording shorter than one processing window')
    if c['nfft']!=n or c['range_bin_start']!=0:raise ValueError('This validated runner expects nfft=n_samples and range_bin_start=0')
    angles=np.arange(c['angle_min_deg'],c['angle_max_deg']+c['angle_step_deg']/2,c['angle_step_deg']);bins=np.arange(c['range_bin_stop']);ranges=bins*c['sample_rate_hz']/n*3e8/(2*c['slope_hz_s']);starts=np.arange(0,observations-window+1,hop)
    steering=steering_matrix(ranges,angles,c['spatial_window'],c['fast_window'],c['carrier_hz'],c['slope_hz_s']*100e-6,100e-6,c['sample_rate_hz'],c['antenna_spacing_m'],angular_sign=c['angular_sign'])
    x,_=CPUIOEngine(nr,observations,n).read_frame(str(args.input));tracker=ConfidenceTracker();spectra=[];powers=[];eigs=[];states=[];rows=[]
    print(test_tracker(),flush=True)
    for k,lo in enumerate(starts):
        raw=x[:,lo:lo+window];scene=raw-raw.mean(axis=2,keepdims=True);motion=scene-scene.mean(axis=1,keepdims=True)
        rp=np.fft.fft(motion*np.hanning(n),axis=-1)[:,:,:len(bins)];power=np.mean(abs(rp)**2,axis=(0,1))
        cov=smoothed_covariance(motion,c['spatial_window'],c['fast_window'],c['chirp_step'],c['fast_step'],c['forward_backward']);sp,ev=spectrum_from_steering(cov,steering,(len(angles),len(bins)),c['signal_rank']);z=10*np.log10(np.maximum(sp/sp.max(),1e-12));ds=candidates(z,power,angles);visible,lost=tracker.step(ds,k)
        spectra.append(z);powers.append(power);eigs.append(ev);states.append(dict(candidates=ds,visible=visible,lost=lost,map_contrast_db=float(z.max()-np.median(z[:,8:]))))
        for t in visible:rows.append(dict(frame=k,start_observation=int(lo),**t))
        if k%10==0:print(f'Processed {k+1}/{len(starts)}',flush=True)
    ts=[t for t in tracker.all() if t['confirmed']];summary=dict(input_name=args.input.name,input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),observations=observations,windows=len(starts),signal_rank=c['signal_rank'],candidates=sum(len(s['candidates']) for s in states),confirmed_windows=sum(any(t['confirmed'] for t in s['visible']) for s in states),confirmed_track_segments=len(ts),tracks=[dict(id=t['id'],history=t['history']) for t in ts],tail_observations_not_in_complete_window=int(observations-(starts[-1]+window)))
    arrays=dict(joint=np.array(spectra),range_power=np.array(powers),eigenvalues=np.array(eigs),angles=angles,range_bins=bins,starts=starts,window_observations=window)
    np.savez_compressed(out/'maps.npz',**arrays);(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'frame_states.json').write_text(json.dumps(states,indent=2));(out/'settings.json').write_text(json.dumps(dict(radar=c,detector_tracker=SETTINGS),indent=2))
    with (out/'reflection_tracks.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['frame','start_observation','id','range_bin','angle_deg','confirmed','support']);w.writeheader();w.writerows(rows)
    states,summary=apply_grouping(out,arrays,states,summary,c.get('expected_people',1));render(out,arrays,states,summary);render(out,arrays,states,summary,debug=True);print('Complete:',out,flush=True)
if __name__=='__main__':main()
