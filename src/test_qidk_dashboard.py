import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import qidk_dashboard_runtime as runtime
from heterogeneous import HeterogeneousPipeline


class QidkDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        runtime.db_path = Path(self.temp_dir.name) / 'results.sqlite3'
        runtime.initialize_database()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_batch_pipeline_reports_bad_capture_without_stalling(self):
        pipeline = HeterogeneousPipeline(
            m_chirps=4, n_chirps=4, n_fft_range=4, azimuth_grid=[30, 0, -30]
        )
        results, _ = pipeline.process_batch_asynchronous([
            str(Path(self.temp_dir.name) / 'missing.bin')
        ])
        self.assertEqual(len(results), 1)
        self.assertIn('CPU ingest failed:', results[0]['error'])
        self.assertGreater(results[0]['timing']['total_e2e_ms'], 0)

    def test_valid_capture_runs_through_all_pipeline_stages(self):
        capture = Path(self.temp_dir.name) / 'capture.bin'
        np.zeros(2 * 4 * 64 * 64, dtype=np.int16).tofile(capture)
        pipeline = HeterogeneousPipeline(
            m_chirps=64, n_chirps=64, n_fft_range=64,
            azimuth_grid=[30, 0, -30], fs_vital=20,
        )
        results, _ = pipeline.process_batch_asynchronous([str(capture)])
        self.assertEqual(len(results), 1)
        self.assertNotIn('error', results[0])
        self.assertGreater(results[0]['timing']['total_e2e_ms'], 0)
        self.assertEqual(results[0]['ra_heatmap'].shape, (3, 32))
        self.assertIn('gpu', results[0]['backends'])
        self.assertIn('npu', results[0]['backends'])

    def test_generated_plots_and_results_survive_database_round_trip(self):
        run_id = 'test-run'
        frame = {
            'file': '/tmp/D1.0m_0d_HR_70_pm_5_BR_15.0.bin',
            'range_m': 1.0,
            'angle_deg': 0.0,
            'hr_fft': 70.0,
            'br_fft': 15.0,
            'hr_ac': 69.0,
            'br_ac': 14.0,
            'timing': {
                'cpu_read_ms': 1.2,
                'gpu_total_ms': 3.4,
                'npu_detect_ms': 0.5,
                'cpu_vitals_total_ms': 1.1,
                'total_e2e_ms': 6.5,
            },
            'backends': {'gpu': 'NumPy CPU (BLAS)', 'npu': 'CPU fallback', 'cpu': 'CPU'},
            'disp_mm': np.sin(np.arange(512) / 10),
            'resp_signal': np.sin(np.arange(512) / 20),
            'heart_signal': np.sin(np.arange(512) / 5),
            'ra_heatmap': np.random.default_rng(1).random((73, 256)),
        }
        class FakePipeline:
            def process_batch_asynchronous(self, paths, callback):
                frame['file'] = paths[0]
                callback(frame)
                return [frame], 0.1

        with runtime.state_lock:
            runtime.run_state.update(run_id=run_id, state='running', done=0, total=1)
        with patch.object(runtime, 'HeterogeneousPipeline', FakePipeline):
            runtime.run_pipeline([Path(frame['file'])], run_id)

        client = runtime.app.test_client()
        state = client.get('/api/state').get_json()
        self.assertEqual(state['results'][0]['result']['timing']['gpu_total_ms'], 3.4)
        self.assertEqual(state['state'], 'done')
        for kind in ('heatmap', 'displacement', 'spectra'):
            response = client.get(state['results'][0]['plots'][kind])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'image/png')
        data_response = client.get(state['results'][0]['data'])
        self.assertEqual(data_response.status_code, 200)
        with np.load(io.BytesIO(data_response.data)) as analysis:
            self.assertEqual(analysis['range_azimuth_heatmap'].shape, (73, 256))


if __name__ == '__main__':
    unittest.main()
