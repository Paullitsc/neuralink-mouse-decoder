
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
   b) Chamfer distance to hand-crafted prototypes (A, N, K, E, I, O, Q, D, R, P, B, M, W, H, L, T, U, V, X, Y, Z)
   c) Small search over mirroring and rotation (±20°) for robustness.
4) Print the predicted string, and optionally run the provided checker.

Usage:
  python decode_mouse_velocities.py --csv mouse_velocities.csv
  python decode_mouse_velocities.py --csv mouse_velocities.csv --check ./check_answer.py
  python decode_mouse_velocities.py --self_train --clusters 8 --blend_weight 0.6

Notes:
- This is fully algorithmic; no deep learning.
- Chamfer distance uses a simple O(N*M) nearest-distance - fast at our scales.
- Parameter-free pause detection using Otsu thresholding.
- Enhanced letter disambiguation via RDP simplification and vertical-extrema analysis.
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
    """Load CSV with required columns: timestamp, velocity_x, velocity_y."""
    df = pd.read_csv(path)
    assert {'timestamp','velocity_x','velocity_y'}.issubset(df.columns), "CSV must have timestamp, velocity_x, velocity_y"
    return df

def reconstruct_positions(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct x,y positions by integrating velocity deltas. Y-axis is flipped so up is positive."""
    vx = df['velocity_x'].to_numpy(dtype=float)
    vy = df['velocity_y'].to_numpy(dtype=float)
    x = np.cumsum(vx)
    y = -np.cumsum(vy)   # flip Y so up is positive
    return x, y

def otsu_threshold(speeds: np.ndarray) -> float:
    """Find optimal speed threshold using Otsu's method for bimodal distribution."""
    if len(speeds) < 2:
        return np.median(speeds)
    
    # Use histogram with reasonable number of bins
    bins = min(50, len(speeds) // 10)
    hist, bin_edges = np.histogram(speeds, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Otsu's method: maximize between-class variance
    total_pixels = len(speeds)
    total_mean = np.mean(speeds)
    
    best_threshold = bin_centers[0]
    max_variance = 0
    
    for i in range(len(hist)):
        # Split into two classes
        class1_pixels = np.sum(hist[:i+1])
        class2_pixels = total_pixels - class1_pixels
        
        if class1_pixels == 0 or class2_pixels == 0:
            continue
            
        # Calculate means for each class
        class1_mean = np.sum(hist[:i+1] * bin_centers[:i+1]) / class1_pixels
        class2_mean = np.sum(hist[i+1:] * bin_centers[i+1:]) / class2_pixels
        
        # Calculate between-class variance
        variance = class1_pixels * class2_pixels * (class1_mean - class2_mean) ** 2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = bin_centers[i]
    
    return best_threshold

def largest_gap_threshold(speeds: np.ndarray) -> float:
    """Find speed threshold by identifying the largest gap in sorted speed values."""
    if len(speeds) < 2:
        return np.median(speeds)
    
    sorted_speeds = np.sort(speeds)
    gaps = np.diff(sorted_speeds)
    max_gap_idx = np.argmax(gaps)
    
    # Threshold is midpoint of the largest gap
    threshold = (sorted_speeds[max_gap_idx] + sorted_speeds[max_gap_idx + 1]) / 2
    return threshold

def auto_pause_threshold(speeds: np.ndarray, method: str = 'otsu') -> float:
    """Automatically determine pause threshold using specified method."""
    if method == 'otsu':
        return otsu_threshold(speeds)
    elif method == 'gap':
        return largest_gap_threshold(speeds)
    else:
        # Fallback to percentile-based method
        return np.percentile(speeds, 25)

def segment_strokes(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                    method: str = 'auto', min_pause: int = 5, min_len: int = 80) -> List[Tuple[int,int]]:
    """
    Segment strokes using automatic pause detection.
    
    Args:
        method: 'auto', 'otsu', 'gap', or 'percentile'
        min_pause: minimum consecutive low-speed samples to count as pause
        min_len: minimum stroke length in samples
    """
    speeds = np.hypot(vx, vy)
    
    # Auto-detect threshold
    if method == 'auto':
        # Try Otsu first, fallback to gap method
        try:
            threshold = otsu_threshold(speeds)
        except:
            threshold = largest_gap_threshold(speeds)
    else:
        threshold = auto_pause_threshold(speeds, method)
    
    print(f"Auto-detected pause threshold: {threshold:.4f} (method: {method})")
    
    # Segment using detected threshold
    low = speeds < threshold
    pauses = []
    start = None
    
    for i, val in enumerate(low):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_pause:
                pauses.append((start, i-1))
            start = None
    
    if start is not None and len(speeds) - start >= min_pause:
        pauses.append((start, len(speeds)-1))

    # Build alternating runs of "active" segments between pauses
    n = len(speeds)
    cuts = [0, n]
    for s, e in pauses:
        cuts.extend([s, e+1])
    cuts = sorted(set(cuts))

    intervals = []
    for i in range(0, len(cuts)-1, 2):
        s = cuts[i]; e = cuts[i+1]
        if e - s >= min_len:
            intervals.append((s, e))
    
    return intervals

def rdp_simplify(points: np.ndarray, epsilon: float = 0.02) -> np.ndarray:
    """
    Simplify polyline using Ramer-Douglas-Peucker algorithm.
    
    Args:
        points: Nx2 array of points
        epsilon: tolerance for simplification
    """
    if len(points) <= 2:
        return points
    
    def perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment."""
        if np.allclose(line_start, line_end):
            return np.linalg.norm(point - line_start)
        
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        # Vector from line_start to point
        point_vec = point - line_start
        
        # Project point_vec onto line_vec
        t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        t = np.clip(t, 0, 1)
        
        # Closest point on line
        closest = line_start + t * line_vec
        
        return np.linalg.norm(point - closest)
    
    def rdp_recursive(points, epsilon):
        if len(points) <= 2:
            return points
        
        # Find point with maximum distance
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            dist = perpendicular_distance(points[i], points[0], points[-1])
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If max distance is greater than epsilon, recursively simplify
        if max_dist > epsilon:
            left = rdp_recursive(points[:max_idx + 1], epsilon)
            right = rdp_recursive(points[max_idx:], epsilon)
            return np.vstack([left[:-1], right])
        else:
            return np.array([points[0], points[-1]])
    
    return rdp_recursive(points, epsilon)

def find_vertical_extrema(points: np.ndarray) -> Dict[str, List[int]]:
    """
    Find vertical extrema (peaks and valleys) in a stroke.
    
    Returns:
        Dictionary with 'peaks' and 'valleys' lists of indices
    """
    if len(points) < 3:
        return {'peaks': [], 'valleys': []}
    
    y_coords = points[:, 1]
    extrema = {'peaks': [], 'valleys': []}
    
    for i in range(1, len(y_coords) - 1):
        if y_coords[i] > y_coords[i-1] and y_coords[i] > y_coords[i+1]:
            extrema['peaks'].append(i)
        elif y_coords[i] < y_coords[i-1] and y_coords[i] < y_coords[i+1]:
            extrema['valleys'].append(i)
    
    return extrema

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
    """Normalize points to fit in unit box [0,1] x [0,1]."""
    pts = np.asarray(points, dtype=float)
    minv = pts.min(axis=0); maxv = pts.max(axis=0)
    size = max((maxv - minv).max(), 1e-9)
    return (pts - minv) / size

def rotate_around_center(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate points around their center by given angle in degrees."""
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
    dists2 = np.sqrt(((B[:,None,:] - A[None,:,:])**2).sum(axis=1)).min(axis=1)
    return float(dists1.mean() + dists2.mean())


# -------------------- Feature extraction --------------------

@dataclass
class StrokeFeatures:
    width: float
    height: float
    aspect: float
    closure_ratio: float
    angle_hist: np.ndarray  # [horiz, diag, vert] aggregate
    rdp_points: int  # number of points after RDP simplification
    vertical_extrema: Dict[str, List[int]]  # peaks and valleys

def stroke_features(xs: np.ndarray, ys: np.ndarray) -> StrokeFeatures:
    """Extract comprehensive stroke features including RDP simplification and vertical extrema."""
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
    
    # RDP simplification
    points = np.c_[xs, ys]
    rdp_simplified = rdp_simplify(points, epsilon=0.02)
    rdp_count = len(rdp_simplified)
    
    # Vertical extrema
    extrema = find_vertical_extrema(rdp_simplified)
    
    return StrokeFeatures(
        width=w, height=h, aspect=(h+1e-9)/(w+1e-9), 
        closure_ratio=closure, angle_hist=hist,
        rdp_points=rdp_count, vertical_extrema=extrema
    )


# -------------------- Enhanced Prototypes --------------------

def build_prototypes() -> dict:
    """Build comprehensive set of letter prototypes including new additions."""
    def res(points, n): return normalize_unit_box(resample_polyline(np.array(points,float), n))
    
    # Original prototypes
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
    
    # New prototypes
    def proto_R():
        v=[[0.1,0.05],[0.1,0.95]]; arc=[[0.1,0.5],[0.9,0.5],[0.9,0.05]]; leg=[[0.7,0.5],[0.9,0.95]]
        return np.vstack([res(v,100), res(arc,120), res(leg,80)])
    
    def proto_P():
        v=[[0.1,0.05],[0.1,0.95]]; arc=[[0.1,0.5],[0.9,0.5],[0.9,0.05]]
        return np.vstack([res(v,100), res(arc,120)])
    
    def proto_B():
        v=[[0.1,0.05],[0.1,0.95]]; top_arc=[[0.1,0.8],[0.8,0.8],[0.8,0.6]]; bot_arc=[[0.1,0.2],[0.8,0.2],[0.8,0.4]]
        return np.vstack([res(v,100), res(top_arc,100), res(bot_arc,100)])
    
    def proto_M():
        left=[[0.1,0.95],[0.1,0.05]]; peak=[[0.1,0.05],[0.5,0.6],[0.9,0.05]]; right=[[0.9,0.05],[0.9,0.95]]
        return np.vstack([res(left,100), res(peak,120), res(right,100)])
    
    def proto_W():
        left=[[0.1,0.05],[0.1,0.95]]; valley=[[0.1,0.05],[0.3,0.6],[0.5,0.05],[0.7,0.6],[0.9,0.05]]; right=[[0.9,0.05],[0.9,0.95]]
        return np.vstack([res(left,100), res(valley,150), res(right,100)])
    
    def proto_H():
        left=[[0.1,0.05],[0.1,0.95]]; bar=[[0.1,0.5],[0.9,0.5]]; right=[[0.9,0.05],[0.9,0.95]]
        return np.vstack([res(left,100), res(bar,100), res(right,100)])
    
    def proto_L():
        v=[[0.1,0.05],[0.1,0.95]]; h=[[0.1,0.95],[0.9,0.95]]
        return np.vstack([res(v,100), res(h,100)])
    
    def proto_T():
        h=[[0.1,0.05],[0.9,0.05]]; v=[[0.5,0.05],[0.5,0.95]]
        return np.vstack([res(h,100), res(v,100)])
    
    def proto_U():
        left=[[0.1,0.05],[0.1,0.5]]; arc=[[0.1,0.5],[0.9,0.5],[0.9,0.05]]; right=[[0.9,0.05],[0.9,0.5]]
        return np.vstack([res(left,80), res(arc,120), res(right,80)])
    
    def proto_V():
        left=[[0.1,0.05],[0.5,0.95]]; right=[[0.5,0.95],[0.9,0.05]]
        return np.vstack([res(left,100), res(right,100)])
    
    def proto_X():
        diag1=[[0.1,0.05],[0.9,0.95]]; diag2=[[0.1,0.95],[0.9,0.05]]
        return np.vstack([res(diag1,100), res(diag2,100)])
    
    def proto_Y():
        left=[[0.1,0.05],[0.5,0.5]]; right=[[0.5,0.5],[0.9,0.05]]; v=[[0.5,0.5],[0.5,0.95]]
        return np.vstack([res(left,80), res(right,80), res(v,100)])
    
    def proto_Z():
        h1=[[0.1,0.05],[0.9,0.05]]; diag=[[0.9,0.05],[0.1,0.95]]; h2=[[0.1,0.95],[0.9,0.95]]
        return np.vstack([res(h1,100), res(diag,100), res(h2,100)])
    
    return {
        'A': proto_A(), 'N': proto_N(), 'K': proto_K(), 'E': proto_E(), 'I': proto_I(), 
        'O': proto_O(), 'Q': proto_Q(), 'D': proto_D(), 'R': proto_R(), 'P': proto_P(),
        'B': proto_B(), 'M': proto_M(), 'W': proto_W(), 'H': proto_H(), 'L': proto_L(),
        'T': proto_T(), 'U': proto_U(), 'V': proto_V(), 'X': proto_X(), 'Y': proto_Y(), 'Z': proto_Z()
    }


# -------------------- Enhanced Classification --------------------

def classify_stroke(xs: np.ndarray, ys: np.ndarray, PROTOS: dict) -> Tuple[str, float, StrokeFeatures]:
    """Enhanced classification using RDP simplification and vertical-extrema analysis."""
    # normalize and resample
    xs = xs - xs.min(); ys = ys - ys.min()
    pts = np.c_[xs, ys]
    pts = normalize_unit_box(pts)
    pts = resample_polyline(pts, 300)

    feats = stroke_features(xs, ys)

    # Enhanced disambiguation using RDP and vertical extrema
    # 1) Clear vertical line ('I'): tall & skinny, vertical-dominated, not closed
    if feats.aspect > 2.2 and feats.angle_hist[2] > 0.55 and feats.closure_ratio > 0.08:
        return 'I', 0.0, feats
    
    # 2) A/N/K disambiguation using vertical extrema and RDP
    if feats.rdp_points <= 8 and len(feats.vertical_extrema['peaks']) >= 1:
        # Likely A, N, or K - use extrema analysis
        if len(feats.vertical_extrema['peaks']) == 1 and len(feats.vertical_extrema['valleys']) == 0:
            # Single peak - likely A
            return 'A', 0.0, feats
        elif len(feats.vertical_extrema['peaks']) == 0 and len(feats.vertical_extrema['valleys']) == 1:
            # Single valley - likely N
            return 'N', 0.0, feats
        elif len(feats.vertical_extrema['peaks']) == 1 and len(feats.vertical_extrema['valleys']) == 1:
            # Peak and valley - likely K
            return 'K', 0.0, feats
    
    # 3) 'E': strong horizontal + some vertical, not closed
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


# -------------------- Lightweight self-training (writer adaptation) --------------------

def extract_stroke_points(x: np.ndarray, y: np.ndarray, interval: Tuple[int,int]) -> np.ndarray:
    """Extract and normalize stroke points from interval."""
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
    """Map cluster centroids to nearest letter prototypes."""
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
    """Build writer-adapted prototypes by clustering strokes and mapping to letters."""
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
    """Blend base and adapted prototypes with weight w."""
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


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Decode handwritten text from mouse velocity data")
    parser.add_argument('--csv', default='mouse_velocities.csv', help='Path to CSV file')
    parser.add_argument('--check', default=None, help='Path to check_answer.py to verify the decoded string')
    parser.add_argument('--self_train', action='store_true', help='Enable writer adaptation via clustering')
    parser.add_argument('--clusters', type=int, default=6, help='Number of clusters for self-training')
    parser.add_argument('--blend_weight', type=float, default=0.5, help='Blend weight for adapted prototypes [0..1]')
    parser.add_argument('--verbose_self', action='store_true', help='Print self-training diagnostics')
    parser.add_argument('--pause_method', choices=['auto', 'otsu', 'gap', 'percentile'], default='auto', 
                       help='Method for automatic pause detection')
    args = parser.parse_args()

    print(f"Loading CSV: {args.csv}")
    df = load_csv(args.csv)
    x, y = reconstruct_positions(df)
    vx = df['velocity_x'].to_numpy(float)
    vy = df['velocity_y'].to_numpy(float)

    print(f"Reconstructed trajectory: {len(x)} points")
    intervals = segment_strokes(x, y, vx, vy, method=args.pause_method, speed_thresh=0.3, min_pause=5, min_len=80)
    print(f"Segmented into {len(intervals)} strokes")

    PROTOS = build_prototypes()
    if args.self_train:
        print("Building writer-adapted prototypes...")
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
    print('\nDECODED:', decoded)
    print()
    
    # Print detailed analysis
    for i,(ch,score,feats) in enumerate(details,1):
        extrema_info = f"extrema(p{len(feats.vertical_extrema['peaks'])},v{len(feats.vertical_extrema['valleys'])})"
        print(f'{i:2d}. {ch:>2}  score={score:7.3f}  aspect={feats.aspect:5.2f}  closed?={feats.closure_ratio<0.08}  rdp={feats.rdp_points:3d}  {extrema_info}')

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
