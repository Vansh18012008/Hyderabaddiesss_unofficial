"""
heterogeneous package
=====================
Tri-Engine Heterogeneous Execution Framework (CPU + GPU + NPU)
for FMCW Radar Vital Sign Extraction.
"""

from .gpu_engine import GPURadarEngine
from .npu_engine import NPURadarEngine
from .cpu_io_engine import CPUIOEngine
from .cpu_vitals_engine import CPUPostProcessEngine
from .pipeline_manager import HeterogeneousPipeline

__all__ = [
    'GPURadarEngine',
    'NPURadarEngine',
    'CPUIOEngine',
    'CPUPostProcessEngine',
    'HeterogeneousPipeline'
]
