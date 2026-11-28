"""Networked dashboard, local persistence, and pipelined QIDK runner."""

import argparse
import io
import json
import os
import queue
import socket
import sqlite3
import threading
import time
import uuid
import webbrowser
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, jsonify, render_template_string, send_file

import qidk_web_server as dashboard
from heterogeneous import HeterogeneousPipeline


app = Flask(__name__)
db_path = dashboard.DB_PATH
state_lock = threading.Lock()
event_lock = threading.Lock()
client_queues = []
event_history = []
run_state = {
    'state': 'idle',
    'message': 'Starting dashboard',
    'run_id': None,
    'done': 0,
    'total': 0,
    'failures': 0,
}
PLOT_COLUMNS = {
    'heatmap': 'heatmap_png',
    'displacement': 'displacement_png',
    'spectra': 'spectra_png',
}


@contextmanager
def connect_db():
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=30)
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize_database():
    with connect_db() as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                meta_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                hr_error REAL,
                br_error REAL,
                error TEXT,
                heatmap_png BLOB,
                displacement_png BLOB,
                spectra_png BLOB,
                analysis_npz BLOB
            )
        ''')
        columns = {row[1] for row in db.execute('PRAGMA table_info(results)')}
        if 'analysis_npz' not in columns:
            db.execute('ALTER TABLE results ADD COLUMN analysis_npz BLOB')


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__} to JSON')


def store_result(run_id, filename, meta, result, error=None, plots=None, analysis_npz=None):
    plots = plots or {}
    hr_error = (
        round(abs(result['hr_fft'] - meta['hr_true']), 2)
        if not error and meta.get('hr_true') else None
    )
    br_error = (
        round(abs(result['br_fft'] - meta['br_true']), 2)
        if not error and meta.get('has_br') else None
    )
    with connect_db() as db:
        cursor = db.execute(
            '''INSERT INTO results
               (run_id, filename, meta_json, result_json, hr_error, br_error,
                error, heatmap_png, displacement_png, spectra_png, analysis_npz)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                run_id,
                filename,
                json.dumps(meta),
                json.dumps(result, default=json_default),
                hr_error,
                br_error,
                error,
                plots.get('heatmap'),
                plots.get('displacement'),
                plots.get('spectra'),
                analysis_npz,
            ),
        )
        return cursor.lastrowid, hr_error


def public_row(row):
    result_id, filename, meta_json, result_json, error = row
    return {
        'id': result_id,
        'filename': filename,
        'meta': json.loads(meta_json),
        'result': json.loads(result_json),
        'error': error,
        'plots': {kind: f'/plots/{result_id}/{kind}' for kind in PLOT_COLUMNS},
        'data': f'/data/{result_id}',
    }


def publish(event_type, data):
    event = {'type': event_type, 'data': data}
    with event_lock:
        event_history.append(event)
        del event_history[:-200]
        for client_queue in client_queues:
            client_queue.put(event)


def prepare_result(frame, run_id):
    filename = Path(frame['file']).name
    meta = dashboard.parse_meta(frame['file'])
    if frame.get('error'):
        result_id, _ = store_result(
            run_id, filename, meta, {'timing': frame.get('timing', {})},
            error=frame['error'],
        )
        return {
            'id': result_id, 'filename': filename, 'meta': meta,
            'error': frame['error'], 'result': {}, 'plots': {},
        }

    br_hz, br_freqs, br_spec = dashboard.fft_peak(
        frame['resp_signal'], dashboard.FS_VITAL, 0.1, 0.5
    )
    hr_hz, hr_freqs, hr_spec = dashboard.fft_peak(
        frame['heart_signal'], dashboard.FS_VITAL, 0.8, 2.5
    )
    result = {
        'rng_m': round(float(frame['range_m']), 3),
        'ang_d': round(float(frame['angle_deg']), 2),
        'hr_fft': round(float(frame['hr_fft']), 1),
        'br_fft': round(float(frame['br_fft']), 1),
        'hr_ac': frame['hr_ac'],
        'br_ac': frame['br_ac'],
        'elapsed_ms': round(float(frame['timing']['total_e2e_ms']), 2),
        'timing': frame['timing'],
        'backends': frame['backends'],
    }
    plot_data = {
        'rng_m': result['rng_m'],
        'ang_d': result['ang_d'],
        'hr_fft': result['hr_fft'],
        'br_fft': result['br_fft'],
        'disp_mm': frame['disp_mm'],
        'resp_signal': frame['resp_signal'],
        'heart_signal': frame['heart_signal'],
        'ra_heatmap': frame['ra_heatmap'],
        'br_hz': br_hz,
        'br_freqs': br_freqs,
        'br_spec': br_spec,
        'hr_hz': hr_hz,
        'hr_freqs': hr_freqs,
        'hr_spec': hr_spec,
    }
    plots = {
        'heatmap': dashboard.make_heatmap_fig(plot_data),
        'displacement': dashboard.make_displacement_fig(plot_data),
        'spectra': dashboard.make_spectra_fig(plot_data, meta),
    }
    analysis = io.BytesIO()
    np.savez_compressed(
        analysis,
        range_azimuth_heatmap=frame['ra_heatmap'],
        range_axis_m=dashboard.RANGE_AXIS[:frame['ra_heatmap'].shape[1]],
        azimuth_axis_deg=dashboard.AZIMUTH_GRID,
        displacement_mm=frame['disp_mm'],
        respiration_signal=frame['resp_signal'],
        heart_signal=frame['heart_signal'],
        respiration_frequency_hz=br_freqs,
        respiration_spectrum=br_spec,
        heart_frequency_hz=hr_freqs,
        heart_spectrum=hr_spec,
    )
    result_id, hr_error = store_result(
        run_id, filename, meta, result, plots=plots, analysis_npz=analysis.getvalue()
    )
    return {
        'id': result_id,
        'filename': filename,
        'meta': meta,
        'result': result,
        'hr_err': hr_error,
        'plots': {kind: f'/plots/{result_id}/{kind}' for kind in plots},
        'data': f'/data/{result_id}',
    }


def output_worker(output_queue, stop_token, run_id, total):
    frame_count = 0
    failures = 0
    hr_errors = []
    while True:
        frame = output_queue.get()
        try:
            if frame is stop_token:
                return frame_count, failures, hr_errors
            filename = Path(frame['file']).name
            try:
                event_data = prepare_result(frame, run_id)
            except Exception as exc:
                message = f'Dashboard rendering or database write failed: {exc}'
                result_id, _ = store_result(
                    run_id, filename, dashboard.parse_meta(frame['file']),
                    {'timing': frame.get('timing', {})}, error=message,
                )
                event_data = {
                    'id': result_id, 'filename': filename,
                    'meta': dashboard.parse_meta(frame['file']), 'error': message,
                    'result': {}, 'plots': {},
                }
            frame_count += 1
            if event_data.get('error'):
                failures += 1
            elif event_data.get('hr_err') is not None:
                hr_errors.append(event_data['hr_err'])
            with state_lock:
                run_state.update(done=frame_count, failures=failures)
            publish('progress', {
                'done': frame_count,
                'total': total,
                'filename': filename,
                'filepath': frame['file'],
            })
            publish('result', event_data)
        finally:
            output_queue.task_done()


def run_pipeline(bin_files, run_id):
    total = len(bin_files)
    if not total:
        with state_lock:
            run_state.update(state='idle', message='Dashboard ready — no matching .bin files found')
            state = dict(run_state)
        publish('status', state)
        return

    with state_lock:
        run_state.update(state='running', message=f'Processing {total} capture(s)', total=total)
        state = dict(run_state)
    publish('status', state)

    output_queue = queue.Queue(maxsize=2)
    stop_token = object()
    writer = threading.Thread(
        target=output_worker,
        args=(output_queue, stop_token, run_id, total),
        name='DashboardWriter',
    )
    writer.start()
    try:
        pipeline = HeterogeneousPipeline()
        _, elapsed_s = pipeline.process_batch_asynchronous(
            [str(path) for path in bin_files],
            callback=output_queue.put,
        )
        output_queue.join()
        output_queue.put(stop_token)
        writer.join()

        with state_lock:
            done = run_state['done']
            failures = run_state['failures']
        with connect_db() as db:
            rows = db.execute(
                'SELECT hr_error FROM results WHERE run_id=? AND hr_error IS NOT NULL',
                (run_id,),
            ).fetchall()
        hr_mae = round(float(np.mean([row[0] for row in rows])), 2) if rows else None
        message = f'Complete — {done} capture(s), {failures} failed'
        with state_lock:
            run_state.update(
                state='done',
                message=message,
                total=total,
                pipeline_seconds=round(elapsed_s, 3),
            )
            state = dict(run_state)
        publish('done', {
            'n_processed': done,
            'n_failed': failures,
            'hr_mae': hr_mae,
            'pipeline_seconds': round(elapsed_s, 3),
        })
        publish('status', state)
    except Exception as exc:
        output_queue.join()
        output_queue.put(stop_token)
        writer.join()
        with state_lock:
            run_state.update(state='error', message=f'Pipeline failed: {exc}')
            state = dict(run_state)
        publish('status', state)


@app.route('/')
def index():
    return render_template_string(dashboard.DASHBOARD_HTML)


@app.route('/stream')
def stream():
    client_queue = queue.Queue()
    with event_lock:
        client_queues.append(client_queue)
        replay = list(event_history)

    def generate():
        for event in replay:
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        try:
            while True:
                try:
                    event = client_queue.get(timeout=25)
                    yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                except queue.Empty:
                    yield ': keepalive\n\n'
        finally:
            with event_lock:
                if client_queue in client_queues:
                    client_queues.remove(client_queue)

    return Response(
        generate(), mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/state')
def api_state():
    with state_lock:
        state = dict(run_state)
    with connect_db() as db:
        if state['run_id']:
            rows = db.execute(
                '''SELECT id, filename, meta_json, result_json, error FROM results
                   WHERE run_id=? ORDER BY id DESC LIMIT 100''',
                (state['run_id'],),
            ).fetchall()
        else:
            rows = db.execute(
                '''SELECT id, filename, meta_json, result_json, error FROM results
                   ORDER BY id DESC LIMIT 20'''
            ).fetchall()
    state['results'] = [public_row(row) for row in reversed(rows)]
    return jsonify(state)


@app.route('/results')
def results():
    with connect_db() as db:
        rows = db.execute(
            '''SELECT id, filename, meta_json, result_json, error FROM results
               ORDER BY id DESC LIMIT 500'''
        ).fetchall()
    return jsonify([public_row(row) for row in rows])


@app.route('/plots/<int:result_id>/<kind>')
def plot(result_id, kind):
    column = PLOT_COLUMNS.get(kind)
    if column is None:
        abort(404)
    with connect_db() as db:
        row = db.execute(f'SELECT {column} FROM results WHERE id=?', (result_id,)).fetchone()
    if row is None or row[0] is None:
        abort(404)
    return send_file(io.BytesIO(row[0]), mimetype='image/png', max_age=3600)


@app.route('/data/<int:result_id>')
def analysis_data(result_id):
    with connect_db() as db:
        row = db.execute(
            'SELECT analysis_npz FROM results WHERE id=?', (result_id,)
        ).fetchone()
    if row is None or row[0] is None:
        abort(404)
    return send_file(
        io.BytesIO(row[0]),
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'qidk-analysis-{result_id}.npz',
    )


def local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(('192.0.2.1', 1))
            return sock.getsockname()[0]
    except OSError:
        return '127.0.0.1'


def main():
    global db_path
    parser = argparse.ArgumentParser(description='QIDK FMCW dashboard and pipelined runner')
    parser.add_argument('--file', help='Process a single .bin capture')
    parser.add_argument('--folder', default=str(dashboard.SINGLE_DIR), help='Folder of .bin captures')
    parser.add_argument('--max', type=int, default=None, help='Maximum captures to process')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='0.0.0.0', help='Server bind address')
    parser.add_argument('--db', type=Path, default=dashboard.DB_PATH, help='SQLite file stored on this device')
    parser.add_argument('--no-browser', action='store_true', help='Do not open a local browser')
    args = parser.parse_args()
    if args.max is not None and args.max < 1:
        parser.error('--max must be greater than zero')

    db_path = args.db.expanduser().resolve()
    if args.file:
        bin_files = [Path(args.file).expanduser().resolve()]
    else:
        folder = Path(args.folder).expanduser()
        bin_files = sorted(
            file.resolve() for file in folder.glob('*.bin')
            if dashboard._RE.match(file.name)
        )
        if args.max is not None:
            bin_files = bin_files[:args.max]

    initialize_database()
    run_id = uuid.uuid4().hex
    with state_lock:
        run_state.update(
            state='running' if bin_files else 'idle',
            message=f'Preparing {len(bin_files)} capture(s)' if bin_files
            else 'Dashboard ready — no matching .bin files found',
            run_id=run_id,
            done=0,
            total=len(bin_files),
            failures=0,
        )
    with event_lock:
        event_history.clear()

    address = f'http://{local_ip()}:{args.port}'
    print('\nFMCW QIDK dashboard')
    print(f'Captures : {len(bin_files)}')
    print(f'Dashboard: {address} (same machine: http://localhost:{args.port})')
    print(f'Data     : {db_path}')
    print('Keep this process running; open the LAN address on your PC/QIDK.')
    if os.environ.get('QIDK_NPU_MODEL'):
        print(f'NPU model: {os.environ["QIDK_NPU_MODEL"]}')
    else:
        print('NPU model: not configured; detector uses CPU fallback')
    print('GPU path : CUDA only when PyTorch CUDA is available; otherwise NumPy/CPU\n')

    threading.Thread(target=run_pipeline, args=(bin_files, run_id), daemon=True).start()
    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(1), webbrowser.open(f'http://127.0.0.1:{args.port}')),
            daemon=True,
        ).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
