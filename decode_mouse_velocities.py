
#!/usr/bin/env python3
"""
decode_mouse_velocities.py

Algorithmic recognizer for a Neuralink-style take-home:
- Input: CSV with timestamp, velocity_x, velocity_y sampled every 15ms
- Output: Best-guess uppercase ASCII string decoded from the handwritten strokes

Pipeline:
1) Reconstruct trajectory by integrating velocity deltas.
2) Detect pauses (low-speed runs) to segment the stream into strokes (letters).
3) For each stroke, normalize, resample, and classify using a hybrid:
   a) Fast shape heuristics (closure, aspect, angle histogram)
   b) Chamfer distance to hand-crafted prototypes (A, N, K, E, I, O, Q, D)
   c) Small search over mirroring and rotation (±20°) for robustness.
4) Print the predicted string, and optionally run the provided checker.

Usage:
  python decode_mouse_velocities.py --csv mouse_velocities.csv
  python decode_mouse_velocities.py --csv mouse_velocities.csv --check ./check_answer.py

Notes:
- This is fully algorithmic; no deep learning.
- Chamfer distance uses a simple O(N*M) nearest-distance - fast at our scales.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# -------------------- Utilities --------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {'timestamp','velocity_x','velocity_y'}.issubset(df.columns), "CSV must have timestamp, velocity_x, velocity_y"
    return df

def reconstruct_positions(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    vx = df['velocity_x'].to_numpy(dtype=float)
    vy = df['velocity_y'].to_numpy(dtype=float)
    x = np.cumsum(vx)
    y = -np.cumsum(vy)   # flip Y so up is positive
    return x, y

def segment_strokes(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                    speed_thresh: float = 0.3, min_pause: int = 5, min_len: int = 80) -> List[Tuple[int,int]]:
    """Return [ (start_idx, end_idx) ] intervals for strokes separated by low-speed pauses."""
    speed = np.hypot(vx, vy)
    low = speed < speed_thresh
    pauses = []
    start = None
    for i,val in enumerate(low):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_pause:
                pauses.append((start, i-1))
            start = None
    if start is not None and len(speed) - start >= min_pause:
        pauses.append((start, len(speed)-1))

    # build alternating runs of "active" segments between pauses
    n = len(speed)
    cuts = [0, n]
    for s,e in pauses:
        cuts.extend([s, e+1])
    cuts = sorted(set(cuts))

    intervals = []
    for i in range(0, len(cuts)-1, 2):
        s = cuts[i]; e = cuts[i+1]
        if e - s >= min_len:
            intervals.append((s, e))
    return intervals

def resample_polyline(points: np.ndarray, n: int = 300) -> np.ndarray:
    """Resample a polyline to n points uniformly along arc length."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros((n,2), float)
    d = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0.0, float(s[-1]), n)
    out = np.zeros((n,2), float)
    j = 0
    for i,ti in enumerate(t):
        while j < len(s)-2 and s[j+1] < ti:
            j += 1
        denom = (s[j+1] - s[j])
        w = 0.0 if denom == 0 else (ti - s[j]) / denom
        out[i] = pts[j]*(1-w) + pts[j+1]*w
    return out

def normalize_unit_box(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    minv = pts.min(axis=0); maxv = pts.max(axis=0)
    size = max((maxv - minv).max(), 1e-9)
    return (pts - minv) / size

def rotate_around_center(points: np.ndarray, angle_deg: float) -> np.ndarray:
    ang = math.radians(angle_deg)
    R = np.array([[math.cos(ang), -math.sin(ang)],
                  [math.sin(ang),  math.cos(ang)]], float)
    pts = points - 0.5
    pts = pts @ R.T
    pts += 0.5
    return pts

def chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point sets. O(N*M) but fine for N=M=300."""
    # A->B
    dists1 = np.sqrt(((A[:,None,:] - B[None,:,:])**2).sum(axis=2)).min(axis=1)
    # B->A
    dists2 = np.sqrt(((B[:,None,:] - A[None,:,:])**2).sum(axis=2)).min(axis=1)
    return float(dists1.mean() + dists2.mean())


# -------------------- Feature extraction --------------------

@dataclass
class StrokeFeatures:
    width: float
    height: float
    aspect: float
    closure_ratio: float
    angle_hist: np.ndarray  # [horiz, diag, vert] aggregate

def stroke_features(xs: np.ndarray, ys: np.ndarray) -> StrokeFeatures:
    xs = xs - xs.min(); ys = ys - ys.min()
    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())
    diag = float(math.hypot(w, h) + 1e-9)
    closure = float(math.hypot(xs[-1]-xs[0], ys[-1]-ys[0]) / diag)

    dx = np.diff(xs); dy = np.diff(ys)
    ang = np.arctan2(dy, dx)
    ang = (ang + np.pi) % np.pi  # fold to [0,pi)
    # bins centered near 0 (horiz), pi/4 (diag), pi/2 (vert)
    bins = np.histogram(ang, bins=[0, np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8, np.pi])[0]
    horiz = bins[0] + bins[-1]
    diagc = bins[1] + bins[3]
    vert = bins[2]
    total = max(int(horiz+diagc+vert), 1)
    hist = np.array([horiz, diagc, vert], float) / total
    return StrokeFeatures(width=w, height=h, aspect=(h+1e-9)/(w+1e-9), closure_ratio=closure, angle_hist=hist)


# -------------------- Lightweight self-training (writer adaptation) --------------------

def extract_stroke_points(x: np.ndarray, y: np.ndarray, interval: Tuple[int,int]) -> np.ndarray:
    s, e = interval
    xs = x[s:e]
    ys = y[s:e]
    pts = np.c_[xs - xs.min(), ys - ys.min()]
    pts = normalize_unit_box(pts)
    pts = resample_polyline(pts, 300)
    return pts

def kmeans_points(point_sets: List[np.ndarray], k: int, iters: int = 20, seed: int = 42) -> Tuple[List[np.ndarray], np.ndarray]:
    """Cluster polylines represented as 300x2 arrays using Chamfer distance as metric.
    Returns (centroids, labels)."""
    rng = np.random.default_rng(seed)
    n = len(point_sets)
    k = max(1, min(k, n))
    # Initialize centroids by sampling
    indices = rng.choice(n, size=k, replace=False)
    centroids = [point_sets[i].copy() for i in indices]
    labels = np.zeros(n, dtype=int)

    for _ in range(iters):
        # Assign
        for i, pts in enumerate(point_sets):
            d_best = 1e9; j_best = 0
            for j, c in enumerate(centroids):
                d = chamfer_distance(pts, c)
                if d < d_best:
                    d_best = d; j_best = j
            labels[i] = j_best
        # Update
        new_centroids: List[np.ndarray] = []
        for j in range(k):
            members = [point_sets[i] for i in range(n) if labels[i] == j]
            if len(members) == 0:
                # Re-seed an empty cluster
                new_centroids.append(centroids[j])
                continue
            # Average after aligning via simple Procrustes (center + scale to unit box)
            acc = np.zeros_like(centroids[j])
            for pts in members:
                acc += pts
            mean_pts = acc / float(len(members))
            mean_pts = normalize_unit_box(mean_pts)
            new_centroids.append(mean_pts)
        # Check convergence (optional)
        deltas = [chamfer_distance(a, b) for a, b in zip(centroids, new_centroids)]
        centroids = new_centroids
        if sum(deltas) < 1e-4:
            break
    return centroids, labels

def map_clusters_to_letters(centroids: List[np.ndarray], protos: Dict[str, np.ndarray]) -> Tuple[Dict[int,str], Dict[int,float]]:
    cluster_to_letter: Dict[int,str] = {}
    cluster_to_dist: Dict[int,float] = {}
    for j, c in enumerate(centroids):
        best_ch = None; best_d = 1e9
        for ch, p in protos.items():
            d = chamfer_distance(c, p)
            if d < best_d:
                best_d = d; best_ch = ch
        cluster_to_letter[j] = best_ch if best_ch is not None else '?'
        cluster_to_dist[j] = float(best_d)
    return cluster_to_letter, cluster_to_dist

def build_adapted_prototypes(x: np.ndarray, y: np.ndarray, intervals: List[Tuple[int,int]], base_protos: Dict[str,np.ndarray], k: int, verbose: bool = False) -> Dict[str, np.ndarray]:
    strokes = [extract_stroke_points(x, y, iv) for iv in intervals]
    if len(strokes) == 0:
        return dict(base_protos)
    centroids, labels = kmeans_points(strokes, k=max(1, min(k, len(strokes))))
    cluster_to_letter, cluster_to_dist = map_clusters_to_letters(centroids, base_protos)
    # For each letter, pick the best matching centroid as its adapted prototype
    adapted: Dict[str, np.ndarray] = {}
    for idx, ch in cluster_to_letter.items():
        if ch is None or ch == '?':
            continue
        if ch not in adapted or cluster_to_dist[idx] < chamfer_distance(adapted[ch], base_protos[ch]):
            adapted[ch] = centroids[idx]
    if verbose:
        print('Self-train clusters:')
        counts = {j: int((labels == j).sum()) for j in range(len(centroids))}
        rows = []
        for j in range(len(centroids)):
            rows.append((j, counts[j], cluster_to_letter[j], cluster_to_dist[j]))
        rows.sort(key=lambda r: (-r[1], r[3]))
        for j, cnt, ch, dist in rows:
            print(f'  cluster {j}: size={cnt:3d} -> {ch}  match_d={dist:.3f}')
    return adapted

def blend_prototypes(base: Dict[str,np.ndarray], adapted: Dict[str,np.ndarray], w: float) -> Dict[str,np.ndarray]:
    w = float(min(max(w, 0.0), 1.0))
    out: Dict[str,np.ndarray] = {}
    for ch, proto in base.items():
        if ch in adapted:
            a = adapted[ch]
            n = min(len(proto), len(a))
            P = normalize_unit_box(resample_polyline(proto, n))
            A = normalize_unit_box(resample_polyline(a, n))
            blended = (1.0 - w) * P + w * A
            out[ch] = normalize_unit_box(blended)
        else:
            out[ch] = proto
    return out

# -------------------- Prototypes --------------------

def build_prototypes() -> dict:
    def res(points, n): return normalize_unit_box(resample_polyline(np.array(points,float), n))
    def proto_A():
        leg1=[[0.1,0.95],[0.5,0.05],[0.9,0.95]]
        bar=[[0.3,0.55],[0.7,0.55]]
        return np.vstack([res(leg1,200), res(bar,100)])
    def proto_N():
        left=[[0.1,0.95],[0.1,0.05]]; diag=[[0.1,0.05],[0.9,0.95]]; right=[[0.9,0.95],[0.9,0.05]]
        return np.vstack([res(left,100),res(diag,100),res(right,100)])
    def proto_K():
        left=[[0.1,0.95],[0.1,0.05]]; up=[[0.1,0.5],[0.9,0.05]]; down=[[0.1,0.5],[0.9,0.95]]
        return np.vstack([res(left,100),res(up,100),res(down,100)])
    def proto_E():
        v=[[0.1,0.05],[0.1,0.95]]; t=[[0.1,0.05],[0.9,0.05]]; m=[[0.1,0.5],[0.7,0.5]]; b=[[0.1,0.95],[0.9,0.95]]
        return np.vstack([res(v,100),res(t,80),res(m,70),res(b,80)])
    def proto_I():
        v=[[0.5,0.05],[0.5,0.95]]
        return res(v,300)
    def proto_O():
        th=np.linspace(0,2*np.pi,300); r=0.45; cx=0.5; cy=0.5
        return normalize_unit_box(np.c_[cx+r*np.cos(th), cy+r*np.sin(th)])
    def proto_Q():
        o=proto_O(); tail=res([[0.65,0.7],[0.95,0.95]],60)
        return np.vstack([o,tail])
    def proto_D():
        v=[[0.1,0.05],[0.1,0.95]]
        th=np.linspace(-np.pi/2,np.pi/2,180); r=0.45; cx=0.1; cy=0.5
        arc=np.c_[cx + r*np.cos(th), cy + r*np.sin(th)]
        return np.vstack([res(v,120), normalize_unit_box(arc)])
    return {'A': proto_A(), 'N': proto_N(), 'K': proto_K(), 'E': proto_E(), 'I': proto_I(), 'O': proto_O(), 'Q': proto_Q(), 'D': proto_D()}


# -------------------- Classification --------------------

def classify_stroke(xs: np.ndarray, ys: np.ndarray, PROTOS: dict) -> Tuple[str, float, StrokeFeatures]:
    """Return (best_char, score, features)."""
    # normalize and resample
    xs = xs - xs.min(); ys = ys - ys.min()
    pts = np.c_[xs, ys]
    pts = normalize_unit_box(pts)
    pts = resample_polyline(pts, 300)

    feats = stroke_features(xs, ys)

    # Heuristic prefilter
    # 1) Clear vertical line ('I'): tall & skinny, vertical-dominated, not closed
    if feats.aspect > 2.2 and feats.angle_hist[2] > 0.55 and feats.closure_ratio > 0.08:
        return 'I', 0.0, feats
    # 2) 'E': strong horizontal + some vertical, not closed
    if feats.angle_hist[0] > 0.48 and feats.angle_hist[2] > 0.18 and feats.closure_ratio > 0.05:
        # let chamfer confirm between E/K/A
        pass

    best_ch = None; best_d = 1e9
    for mirror in [False, True]:
        pts_m = pts.copy()
        if mirror:
            pts_m[:,0] = 1.0 - pts_m[:,0]
        for ang in [-20,-10,-5,0,5,10,20]:
            cand = rotate_around_center(pts_m, ang)
            for ch,proto in PROTOS.items():
                d = chamfer_distance(cand, proto)
                if d < best_d:
                    best_d = d; best_ch = ch

    # lightweight tie-breaking with features
    if best_ch in ('O','Q') and feats.closure_ratio > 0.08:
        # not actually closed -> more likely D
        best_ch = 'D'

    return best_ch, float(best_d), feats


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='mouse_velocities.csv')
    parser.add_argument('--check', default=None, help='Path to check_answer.py to verify the decoded string')
    parser.add_argument('--self_train', action='store_true', help='Enable writer adaptation via clustering')
    parser.add_argument('--clusters', type=int, default=6, help='Number of clusters for self-training')
    parser.add_argument('--blend_weight', type=float, default=0.5, help='Blend weight for adapted prototypes [0..1]')
    parser.add_argument('--verbose_self', action='store_true', help='Print self-training diagnostics')
    args = parser.parse_args()

    df = load_csv(args.csv)
    x, y = reconstruct_positions(df)
    vx = df['velocity_x'].to_numpy(float)
    vy = df['velocity_y'].to_numpy(float)

    intervals = segment_strokes(x, y, vx, vy, speed_thresh=0.3, min_pause=5, min_len=80)

    PROTOS = build_prototypes()
    if args.self_train:
        adapted = build_adapted_prototypes(x, y, intervals, PROTOS, k=args.clusters, verbose=args.verbose_self)
        PROTOS = blend_prototypes(PROTOS, adapted, w=args.blend_weight)

    letters = []
    details = []

    for (s,e) in intervals:
        xs = x[s:e]; ys = y[s:e]
        ch, score, feats = classify_stroke(xs, ys, PROTOS)
        letters.append(ch)
        details.append((ch, score, feats))

    decoded = ''.join(ch if ch is not None else '?' for ch in letters)
    print('DECODED:', decoded)
    print()
    for i,(ch,score,feats) in enumerate(details,1):
        print(f'{i:2d}. {ch:>2}  score={score:7.3f}  aspect={feats.aspect:5.2f}  closed?={feats.closure_ratio<0.08}  angles(h,d,v)={feats.angle_hist}')

    if args.check:
        # Delegate to the provided checker
        print('\nRunning checker...')
        import subprocess
        res = subprocess.run([sys.executable, args.check, decoded], capture_output=True, text=True)
        print(res.stdout.strip())
        if res.returncode != 0:
            print('Checker indicates the guess is incorrect.')
        else:
            print('Checker indicates the guess is CORRECT!')

if __name__ == '__main__':
    main()
