"""Recompute rolling BR/HR analysis using saved joint maps and raw BIN."""
from pathlib import Path
import json,numpy as np
from adc_io import CPUIOEngine
from vital_estimation import analyze_vitals
ROOT=Path(__file__).resolve().parent;out=ROOT/'output';cfg=json.loads((ROOT/'config.json').read_text())
with np.load(out/'maps.npz') as a:maps={k:a[k] for k in a.files}
states=json.loads((out/'frame_states.json').read_text());cube,_=CPUIOEngine(cfg['num_rx'],1024,cfg['n_samples']).read_frame(str(ROOT/'data/RATHAN_WALKING_VIDEO.bin'))
rows,summary=analyze_vitals(cube,maps,states,cfg,out);print(json.dumps(summary,indent=2))
