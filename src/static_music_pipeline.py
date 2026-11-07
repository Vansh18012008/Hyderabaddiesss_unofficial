"""Stationary multi-reflector pipeline using the existing per-bin angle MUSIC.

Run from unofficial/src:
 python static_music_pipeline.py --input capture.npz --config radar.json --output result
NPZ input: complex `cube` (RX, observation, fast sample), optional `timestamps_s`.
JSON config fields are defined by RadarConfig. BIN uses the existing DCA1000 parser.
All detections are reflector candidates; vital estimates do not classify humans.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.ndimage import maximum_filter

@dataclass
class RadarConfig:
    sample_rate_hz: float
    slope_hz_s: float
    carrier_hz: float
    antenna_spacing_m: float
    n_samples: int
    num_rx: int = 4
    nfft: int = 512
    angular_sign: int = -1
    min_range_m: float = .4
    max_range_m: float = 6.9
    angle_min_deg: float = -60
    angle_max_deg: float = 60
    angle_step_deg: float = 2.5
    slow_fs_hz: float | None = None
    min_vital_duration_s: float = 20.

class StaticMusicPipeline:
    def __init__(self, config: RadarConfig, method='music', source_dir=None):
        self.cfg=config
        if method not in ('old','cbf_static','music'): raise ValueError('Unknown method')
        self.method=method
        if min(config.sample_rate_hz,config.slope_hz_s,config.carrier_hz,config.antenna_spacing_m,config.angle_step_deg)<=0: raise ValueError('Physical parameters must be positive')
        if config.nfft<config.n_samples or config.num_rx<3 or config.angular_sign not in (-1,1): raise ValueError('Invalid FFT, array size or sign')
        if config.slow_fs_hz is not None and config.slow_fs_hz<=0: raise ValueError('Invalid slow-time sampling rate')
        if config.min_vital_duration_s<20: raise ValueError('Use at least 20 s for the existing vital estimator')
        self.source=Path(source_dir or Path(__file__).parent).resolve()
        sys.path.insert(0,str(self.source))
        def load(name,relative):
            spec=importlib.util.spec_from_file_location(name,self.source/relative)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
        self.music=load('_static_pipeline_music','music_2d.py')
        self.gpu=load('_static_pipeline_gpu','heterogeneous/gpu_engine.py')
        self.io=load('_static_pipeline_io','heterogeneous/cpu_io_engine.py')
        # Future annotations avoid evaluating an existing missing Tuple import on Python <3.14.
        path=self.source/'heterogeneous/cpu_vitals_engine.py'
        namespace={'__name__':'_static_pipeline_vitals'}
        exec(compile('from __future__ import annotations\n'+path.read_text(),str(path),'exec'),namespace)
        self.vital_class=namespace['CPUPostProcessEngine']
        self.angles=np.arange(config.angle_min_deg,config.angle_max_deg+config.angle_step_deg/2,config.angle_step_deg)
        self.ranges=np.arange(config.nfft//2)*config.sample_rate_hz/config.nfft*3e8/(2*config.slope_hz_s)
        self.roi=(self.ranges>=config.min_range_m)&(self.ranges<=config.max_range_m)
        if self.roi.sum()<3: raise ValueError('Range region contains fewer than three bins')

    def _detect(self,z):
        if not np.isfinite(z).all() or np.ptp(z)<3: return []
        # Common comparison detector, applied identically to all three methods.
        mask=(z==maximum_filter(z,size=(3,5)))&(z>=-12)
        candidates=sorted(np.argwhere(mask),key=lambda ij:z[tuple(ij)],reverse=True)
        selected=[]; valid_bins=np.flatnonzero(self.roi)
        for ai,local_ri in candidates:
            if ai in (0,len(self.angles)-1) or local_ri in (0,len(valid_bins)-1): continue
            ri=int(valid_bins[local_ri]); r=float(self.ranges[ri]); a=float(self.angles[ai])
            if any(abs(r-t['range_m'])<=.15 and abs(a-t['angle_deg'])<=8 for t in selected): continue
            selected.append(dict(target_id=len(selected)+1,range_idx=ri,angle_idx=int(ai),range_m=r,angle_deg=a,score_db=float(z[ai,local_ri]),classification='unconfirmed_reflector'))
            if len(selected)==6: break
        return selected

    def _sampling(self,n,timestamps):
        fs=self.cfg.slow_fs_hz
        if timestamps is not None:
            t=np.asarray(timestamps,float)
            if t.shape!=(n,) or not np.isfinite(t).all(): return None,'invalid_timestamps'
            dt=np.diff(t)
            if np.any(dt<=0): return None,'non_increasing_timestamps'
            spacing=float(np.median(dt))
            if np.max(abs(dt-spacing))>spacing*.01: return None,'irregular_timestamps'
            measured=1/spacing
            if fs is not None and abs(measured-fs)/fs>.01: return None,'sampling_rate_mismatch'
            fs=measured
        if fs is None:return None,'unknown_slow_time_sampling'
        if fs<=5:return None,'insufficient_sample_rate_for_heart_band'
        if (n-1)/fs<self.cfg.min_vital_duration_s:return None,'insufficient_observation_duration'
        return fs,'estimated_not_human_classification'

    def process_cube(self,cube,timestamps_s=None):
        start=time.perf_counter();c=self.cfg;x=np.asarray(cube)
        if x.ndim!=3 or x.shape[0]!=c.num_rx or x.shape[2]!=c.n_samples or x.shape[1]<2 or not np.iscomplexobj(x) or not np.isfinite(x).all(): raise ValueError('Expected finite complex (RX, observations, ADC samples)')
        # Original CBF weights have negative phase. Convert display angle convention.
        engine=self.gpu.GPURadarEngine(num_rx=c.num_rx,m_chirps=x.shape[1],n_chirps=c.n_samples,n_fft_range=c.nfft,d_ant=c.antenna_spacing_m,lambda_c=3e8/c.carrier_hz,azimuth_grid=c.angular_sign*self.angles,use_hardware_accel=False)
        rp=None; beamed=None
        if self.method=='old':
            beamed,linear,_=engine.process_frame(x)
            raw=linear[:,self.roi]; z=20*np.log10(np.maximum(raw/max(float(raw.max()),1e-30),1e-12))
        else:
            rp=np.fft.fft(x*np.hanning(c.n_samples),c.nfft,axis=-1)[:,:,:c.nfft//2]
            if self.method=='music':
                # Existing MUSIC assumes negative spatial phase.
                search=-c.angular_sign*self.angles
                spectrum,*_=self.music.compute_music_2d_fast(rp,c.carrier_hz,c.antenna_spacing_m,3e8,3e8/c.carrier_hz,self.ranges,search,m1=3,num_targets=2,num_targets_method='gap')
                z=spectrum[self.roi].T;z=z-z.max()
            else:
                beamed=np.einsum('ra,rcs->acs',engine.cbf_weights,rp)
                raw=np.mean(abs(beamed[:,:,self.roi]),axis=1);z=20*np.log10(np.maximum(raw/max(float(raw.max()),1e-30),1e-12))
        targets=self._detect(z)
        fs,status=self._sampling(x.shape[1],timestamps_s)
        outputs=[];iqs=[];signals={}
        estimator=self.vital_class(fs_vital=fs,lambda_c=3e8/c.carrier_hz) if fs else None
        for t in targets:
            ri,ai=t['range_idx'],t['angle_idx']
            iq=beamed[ai,:,ri] if beamed is not None else engine.cbf_weights[:,ai]@rp[:,:,ri]
            iqs.append(iq); out=dict(t,vitals_status=status,hr_fft=None,br_fft=None,hr_ac=None,br_ac=None)
            if estimator is not None and np.linalg.norm(iq)>0:
                try:
                    vitals,_=estimator.extract_vitals(iq)
                    for k in ('hr_fft','br_fft','hr_ac','br_ac'):
                        value=vitals[k];out[k]=float(value) if value is not None and np.isfinite(value) else None
                    for k in ('disp_mm','resp_signal','heart_signal'): signals[f"target_{t['target_id']}_{k}"]=vitals[k]
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    out['vitals_status']='estimation_failed';out['vitals_error']=str(exc)
            outputs.append(out)
        return dict(method=self.method,targets=outputs,ra_db=z,range_axis=self.ranges[self.roi],angle_axis=self.angles,slow_iq=np.array(iqs) if iqs else np.empty((0,x.shape[1]),complex),signals=signals,vitals_status=status,seconds=time.perf_counter()-start)

    def process_bin(self,path,n_observations):
        expected=2*self.cfg.num_rx*n_observations*self.cfg.n_samples*2
        if Path(path).stat().st_size!=expected: raise ValueError('BIN length does not match explicit configuration')
        reader=self.io.CPUIOEngine(num_rx=self.cfg.num_rx,m_chirps=n_observations,n_chirps=self.cfg.n_samples)
        cube,_=reader.read_frame(str(path));return self.process_cube(cube)

    @staticmethod
    def save(result,output):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'signals.npz',ra_db=result['ra_db'],range_axis=result['range_axis'],angle_axis=result['angle_axis'],slow_iq=result['slow_iq'],**result['signals'])
        meta={k:result[k] for k in ('method','targets','vitals_status','seconds')}
        (output/'result.json').write_text(json.dumps(meta,indent=2,allow_nan=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--method',choices=['old','cbf_static','music'],default='music');parser.add_argument('--observations',type=int);parser.add_argument('--source-dir')
    args=parser.parse_args(); pipeline=StaticMusicPipeline(RadarConfig(**json.loads(Path(args.config).read_text())),args.method,args.source_dir)
    if Path(args.input).suffix=='.bin':
        if not args.observations: parser.error('--observations is required for BIN input')
        result=pipeline.process_bin(args.input,args.observations)
    else:
        with np.load(args.input,allow_pickle=False) as data: result=pipeline.process_cube(data['cube'],data['timestamps_s'] if 'timestamps_s' in data else None)
    pipeline.save(result,args.output);print(json.dumps({'targets':result['targets'],'vitals_status':result['vitals_status']},indent=2))
