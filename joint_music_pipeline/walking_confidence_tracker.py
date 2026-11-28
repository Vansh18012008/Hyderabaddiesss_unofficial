"""Ungated-map candidate detection and kinematic Kalman association.
Pure mathematical multi-target localization: Cartesian spatial clustering,
RCS power-weighted centroiding, and kinematic Kalman filtering with
Mahalanobis validation gating.
"""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components

SETTINGS = dict(
    strong_map_db=-15.,
    strong_range_db=8.,
    strong_contrast_db=8.,
    weak_map_db=-22.,
    weak_range_db=4.,
    weak_contrast_db=6.,
    min_bin=8,
    max_bin=85,
    range_gate=16.,
    angle_gate=15.,
    confirm_hits=3,
    max_missing=2,
    max_weak_run=5,
    max_births_per_window=8,
    max_candidates=24,
    cluster_radius_m=0.04,
    range_res_m=0.005895,
    kalman_q_pos=0.5,
    kalman_q_vel=1.0,
    kalman_r_meas=1.5,
    mahalanobis_gate=9.21,
)


def candidates(z, power, angles, ranges=None):
    s = SETTINGS
    pdb = 10 * np.log10(np.maximum(power, 1e-30))
    floor = np.median(pdb[s['min_bin']:])
    contrast = float(z[:, s['min_bin']:].max() - np.median(z[:, s['min_bin']:]))
    if contrast < s['weak_contrast_db']:
        return []

    # 1. 2D local maxima detection above weak threshold
    mask = (z == maximum_filter(z, size=(3, 7))) & (z >= s['weak_map_db'])
    raw_points = []
    max_b = s.get('max_bin', len(power) - 2)
    for ai, ri in sorted(np.argwhere(mask), key=lambda ij: z[tuple(ij)], reverse=True):
        excess = float(pdb[ri] - floor)
        if ri < s['min_bin'] or ri > max_b or ri >= len(power) - 1 or ai in (0, len(angles) - 1) or excess < s['weak_range_db']:
            continue
        raw_points.append((int(ai), int(ri), float(z[ai, ri]), excess))

    if not raw_points:
        return []

    # 2. Polar -> Cartesian physical coordinates (x, y) in meters
    dr = float(ranges[1] - ranges[0]) if (ranges is not None and len(ranges) > 1) else float(s.get('range_res_m', 0.005895))
    r0 = float(ranges[0]) if ranges is not None and len(ranges) > 0 else 0.0

    ais = np.array([p[0] for p in raw_points])
    ris = np.array([p[1] for p in raw_points])
    scores = np.array([p[2] for p in raw_points])
    excesses = np.array([p[3] for p in raw_points])

    dist_m = r0 + ris * dr
    rad = np.deg2rad(angles[ais])
    xs = dist_m * np.sin(rad)
    ys = dist_m * np.cos(rad)

    # 3. Complete-linkage physical metric clustering: all members within cluster_radius_m
    rad_thresh = float(s.get('cluster_radius_m', 0.04))
    groups = []
    for i in range(len(raw_points)):
        for g in groups:
            if all(np.hypot(xs[i] - xs[j], ys[i] - ys[j]) <= rad_thresh for j in g):
                g.append(i)
                break
        else:
            groups.append([i])

    # 4. For each cluster, compute RCS power-weighted centroid (pulls to torso, ignores limbs)
    cluster_summaries = []
    for g in groups:
        cl_scores = scores[g]
        weights = 10.0 ** (cl_scores / 10.0)
        w_sum = float(np.sum(weights))
        if w_sum <= 0:
            w_sum = 1e-12

        cx = float(np.sum(xs[g] * weights) / w_sum)
        cy = float(np.sum(ys[g] * weights) / w_sum)

        c_range = float(np.hypot(cx, cy))
        c_angle = float(np.rad2deg(np.arctan2(cx, cy)))
        c_bin = int(round((c_range - r0) / dr))
        c_bin = int(np.clip(c_bin, s['min_bin'], len(power) - 2))

        max_score = float(np.max(cl_scores))
        max_excess = float(np.max(excesses[g]))
        strong = bool(max_score >= s['strong_map_db'] and max_excess >= s['strong_range_db'] and contrast >= s['strong_contrast_db'])
        cluster_summaries.append((max_score, dict(
            range_bin=c_bin,
            angle_deg=c_angle,
            score_db=max_score,
            range_excess_db=max_excess,
            strong=strong,
            cx_m=cx,
            cy_m=cy
        )))

    # Sort clusters by highest score
    cluster_summaries.sort(key=lambda item: item[0], reverse=True)
    out = []
    for _, c_dict in cluster_summaries:
        if any(np.hypot(c_dict['cx_m'] - q['cx_m'], c_dict['cy_m'] - q['cy_m']) < rad_thresh for q in out):
            continue
        out.append(c_dict)
        if len(out) >= s['max_candidates']:
            break

    for item in out:
        item.pop('cx_m', None)
        item.pop('cy_m', None)

    return out


class ConfidenceTracker:
    def __init__(self, allow_turns=False):
        self.active = []
        self.archive = []
        self.next_id = 1
        self.allow_turns = allow_turns

    def step(self, dets, k):
        s = SETTINGS
        used = set()
        matched = set()

        # 1. State prediction for all active tracks using constant-velocity model
        preds = {}
        P_preds = {}
        for i, t in enumerate(self.active):
            dt = k - t['last']
            if dt <= 0:
                dt = 1
            h = np.array([r[:3] for r in t['history']])
            is_static = bool(self.allow_turns and t['confirmed'] and len(h) >= 3 and np.ptp(h[:, 1]) <= 2.5 and np.ptp(h[:, 2]) <= 4.0)

            F = np.eye(4)
            if not is_static:
                F[0, 2] = dt
                F[1, 3] = dt
                q_pos = s.get('kalman_q_pos', 0.5) * dt
                q_vel = s.get('kalman_q_vel', 1.0) * dt
            else:
                t['vel'][:] = 0.0
                q_pos = 0.05 * dt
                q_vel = 0.01 * dt

            x = np.array([t['pos'][0], t['pos'][1], t['vel'][0], t['vel'][1]], float)
            x_pred = F @ x
            Q = np.diag([q_pos, q_pos, q_vel, q_vel])
            P = t.get('cov', np.diag([2.0, 2.0, 5.0, 5.0]))
            P_pred = F @ P @ F.T + Q
            preds[i] = x_pred
            P_preds[i] = P_pred

        # 2. Priority matching: strong detections first, then weak continuation for confirmed tracks
        for strong_stage in (True, False):
            ids = [i for i, t in enumerate(self.active) if i not in matched and (strong_stage or (t['confirmed'] and t['weak_run'] < s['max_weak_run']))]
            js = [j for j, d in enumerate(dets) if j not in used and d['strong'] == strong_stage]
            if not ids or not js:
                continue

            cost = np.full((len(ids), len(js)), 1e6)
            H = np.zeros((2, 4))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            R = np.diag([s.get('kalman_r_meas', 1.5), s.get('kalman_r_meas', 1.5)])

            for ii, i in enumerate(ids):
                t = self.active[i]
                dt = k - t['last']
                x_pred = preds[i]
                pred_pos = x_pred[:2]
                P_pred = P_preds[i]
                S = H @ P_pred @ H.T + R
                S_inv = np.linalg.inv(S)

                h = np.array([r[:3] for r in t['history']])
                is_static = bool(self.allow_turns and t['confirmed'] and len(h) >= 3 and np.ptp(h[:, 1]) <= 2.5 and np.ptp(h[:, 2]) <= 4.0)

                max_dr = 3.5 if is_static else s['range_gate'] * dt
                max_da = 5.0 if is_static else s['angle_gate'] * dt
                gate_limit = 6.0 if is_static else float(s.get('mahalanobis_gate', 9.21))

                for jj, j in enumerate(js):
                    if dets[j].get('owner') is not None and dets[j]['owner'] != t['id']:
                        continue
                    m_pos = np.array([dets[j]['range_bin'], dets[j]['angle_deg']], float)
                    delta = m_pos - pred_pos
                    travel = abs(m_pos - t['pos'])

                    within_travel = travel[0] <= max_dr and travel[1] <= max_da
                    d_mahal = float(delta.T @ S_inv @ delta)

                    predicted_ok = (d_mahal <= gate_limit) and (abs(delta[0]) <= max_dr) and (abs(delta[1]) <= max_da)
                    if within_travel and predicted_ok:
                        cost[ii, jj] = np.sqrt(d_mahal)
                    elif within_travel and self.allow_turns and strong_stage and not is_static:
                        cost[ii, jj] = 1.0 + np.linalg.norm(travel / [s['range_gate'] * dt, s['angle_gate'] * dt])

            ii, jj = linear_sum_assignment(cost)
            for a, b in zip(ii, jj):
                if cost[a, b] >= 1e5:
                    continue
                i, j = ids[a], js[b]
                t = self.active[i]
                d = dets[j]
                m_pos = np.array([d['range_bin'], d['angle_deg']], float)
                dt = k - t['last']

                # Kalman state update
                x_pred = preds[i]
                P_pred = P_preds[i]
                S = H @ P_pred @ H.T + R
                K_gain = P_pred @ H.T @ np.linalg.inv(S)
                delta = m_pos - x_pred[:2]
                x_upd = x_pred + K_gain @ delta
                P_upd = (np.eye(4) - K_gain @ H) @ P_pred

                h = np.array([r[:3] for r in t['history']])
                is_static = bool(self.allow_turns and t['confirmed'] and len(h) >= 3 and np.ptp(h[:, 1]) <= 2.5 and np.ptp(h[:, 2]) <= 4.0)

                t['pos'] = m_pos
                t['vel'] = np.zeros(2) if is_static else np.clip(x_upd[2:], [-s['range_gate'], -s['angle_gate']], [s['range_gate'], s['angle_gate']])
                t['cov'] = P_upd
                t['last'] = k
                t['missing'] = 0
                t['streak'] = t['streak'] + 1 if strong_stage else 0
                t['confirmed'] = t['confirmed'] or t['streak'] >= s['confirm_hits']
                t['weak_run'] = 0 if strong_stage else t['weak_run'] + 1
                t['support'] = 'strong' if strong_stage else 'weak'
                t['history'].append([k, float(m_pos[0]), float(m_pos[1]), t['support']])
                used.add(j)
                matched.add(i)

        keep = []
        visible = []
        lost = []
        for i, t in enumerate(self.active):
            if i not in matched:
                t['missing'] += 1
                t['streak'] = 0
            if t['missing']:
                if t['confirmed']:
                    lost.append(t['id'])
            else:
                visible.append(t)
            if t['missing'] > s['max_missing']:
                self.archive.append(t)
            else:
                keep.append(t)
        self.active = keep

        births = 0
        for j, d in enumerate(dets):
            if j in used or not d['strong'] or d.get('owner') is not None or births >= s['max_births_per_window']:
                continue
            pos = np.array([d['range_bin'], d['angle_deg']], float)
            t = dict(
                id=self.next_id,
                pos=pos,
                vel=np.zeros(2),
                cov=np.diag([2.0, 2.0, 5.0, 5.0]),
                last=k,
                missing=0,
                streak=1,
                confirmed=False,
                weak_run=0,
                support='strong',
                history=[[k, float(pos[0]), float(pos[1]), 'strong']]
            )
            self.next_id += 1
            self.active.append(t)
            visible.append(t)
            births += 1

        return [dict(id=t['id'], range_bin=float(t['pos'][0]), angle_deg=float(t['pos'][1]), confirmed=t['confirmed'], support=t['support']) for t in visible], lost

    def all(self):
        return self.archive + self.active


def test_tracker():
    def d(r, strong=True):
        return [dict(range_bin=r, angle_deg=0, strong=strong)]
    t = ConfidenceTracker()
    v, _ = t.step(d(50, False), 0)
    assert not v and not t.active
    for k in range(3):
        v, _ = t.step(d(50 + 2 * k), k + 1)
    assert v[0]['confirmed'] and v[0]['id'] == 1
    v, _ = t.step(d(56, False), 4)
    assert v[0]['id'] == 1 and v[0]['support'] == 'weak'
    v, lost = t.step(d(200, False), 5)
    assert not v and 1 in lost
    t.step([], 6)
    t.step([], 7)
    assert not t.active
    t = ConfidenceTracker()
    for k in range(3):
        t.step(d(50), k)
    for k in range(3, 8):
        v, _ = t.step(d(50, False), k)
        assert v
    v, lost = t.step(d(50, False), 8)
    assert not v and 1 in lost
    t = ConfidenceTracker()
    t.step(d(50), 0)
    v, _ = t.step(d(51, False), 1)
    assert not v and not t.active[0]['confirmed']
    return 'PASS: weak candidates cannot start/confirm tracks; confirmed tracks accept nearby weak measurements; distant jumps rejected; missing tracks retired; weak continuation capped.'
