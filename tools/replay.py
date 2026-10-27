"""Offline replay of a CBRO recording as a discrete-time dynamic system (reads the file through tools/recording.py).

The cull is modelled as a state-space system, one step per frame k:
  input   u_k  the camera's motion since frame k-1: translation |dp| (world units), rotation |dtheta| (degrees)
  state   x_k  every top-level walk entry's class in frame k (filed for the main view, dropped for the sun only, hidden,
               out of the view, leased untouched, absent), and the counts of each class
  output  y_k  the work done in frame k: entries touched, judged, filed, children pushed, registrations, refusals
An ideal incremental culler does work in proportion to the innovation (the entries whose class changed, plus what
it must file every frame anyway); the gap between y_k and that is what a cheaper design can remove. The counterfactual
commands replay other policies against the recorded inputs and states and report the work they would do and every
miss (an entry such a policy would have left out in a frame the recording drew it).

usage: python -I tools/replay.py <file.cbrorec> [command] [options]
  inputs              u_k: the camera's translation and rotation per frame, their distribution
  work                y_k: per-frame work by kind (mean, percentiles), against the input
  states              x_k: class counts, the transition matrix between consecutive frames, persistence (run lengths),
                      the innovation per frame and how much of the touched work re-confirmed an unchanged class;
                      a least-squares fit of innovation and work on the input
  certify [--safety S] [--max-lease N] [--trace-frames N] [--settle N]
                      counterfactual: a per-entry certificate for every entry outside the view (how far outside the
                      view its sphere lies, in world units, against the camera's accumulated motion since: translation
                      plus rotation times distance): the touches such a scheme needs per frame, and its misses
  drops               main-view drops (filed for the sun's cascades only): the chains they start per frame (top
                      entries, children pushed under them, refused registrations), what a block-level drop removes
  children            the engine's child pushes against their top entry's decision: what an inherited decision covers
  parents             what the engine pushes children for: parent types and names, per frame
  moved               walk entries whose bound changed (new slots included): by type and name, by class, how far
  feed [--trace-frames N]
                      the filed and sun-only classes split by why: the trace's turn, a move, a force-visible filing
"""
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recording  # noqa: E402
from recording import np  # noqa: E402

# Walk-entry classes (state x_k). 0 = absent (not offered that frame: its node pruned whole, or not walked).
ABSENT, FILED, DROP, HIDDEN, OUTSIDE, UNNEEDED, CHEAP, LEASED, MARKER, FRUSTUM, OTHER = range(11)
CLASS_NAMES = ["absent", "filed", "sun only", "hidden", "outside", "unneeded", "cheap skip", "leased", "marker", "frustum", "other"]
# The main view's coarse classes: what the next frame must do with the entry.
MACRO = np.array([0, 1, 2, 3, 4, 4, 4, 4, 1, 4, 3], dtype=np.uint8)
MACRO_NAMES = ["absent", "drawn", "sun only", "hidden", "out"]


def origins(rec):
    names = rec.enums["origin"]
    return {n: i for i, n in enumerate(names)}


def paths(rec):
    names = rec.enums["path"]
    return {n: i for i, n in enumerate(names)}


def classify(rec, c):
    """The class of each walk-entry candidate (origins walk skipped / marker / walk top / engine top)."""
    o = origins(rec)
    p = paths(rec)
    flags = c["flags"].astype(np.int64)
    left = (flags & (1 << 9)) != 0
    leased = (flags & ((1 << 13) | (1 << 14))) != 0  # (a cluster's lease, v1.98; an entry's own certificate, v1.99)
    cls = np.full(len(c), OTHER, dtype=np.uint8)
    skipped = c["origin"] == o["walk skipped"]
    cls[skipped & leased] = LEASED
    cls[skipped & ~leased] = CHEAP
    cls[c["origin"] == o["walk marker"]] = MARKER
    top = (c["origin"] == o["walk top"]) | (c["origin"] == o["engine top"])
    cls[top & ~left] = FILED
    path = c["path"]
    cls[top & left & (path == p["main view drop"])] = DROP
    cls[top & left & (path == p["never filed: hidden"])] = HIDDEN
    cls[top & left & (path == p["never filed: out of view"])] = OUTSIDE
    cls[top & left & (path == p["never filed: no view needs it"])] = UNNEEDED
    cls[top & left & (path == p["out of view (engine's frustum test)"])] = FRUSTUM
    return cls


class Replay:
    """The recording's CBRO frames as arrays: inputs per frame, walk entries per frame with a dense object id."""

    def __init__(self, path):
        self.rec = rec = recording.Recording(path)
        cbro = np.flatnonzero(rec.frames["mode"] != 0)
        if len(cbro) == 0:
            raise SystemExit("no CBRO frames in this recording")
        self.frame_index = cbro
        self.frames = rec.frames[cbro]
        self.F = len(cbro)
        # u_k
        eye = self.frames["eye"].astype(np.float64)
        rot = self.frames["rotate"].astype(np.float64).reshape(self.F, 3, 3)
        self.dp = np.zeros(self.F)
        self.dp[1:] = np.linalg.norm(np.diff(eye, axis=0), axis=1)
        rel = np.einsum("kij,kil->kjl", rot[:-1], rot[1:])  # R_{k-1}^T R_k (row convention: either way the angle is the same)
        cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
        self.dtheta = np.zeros(self.F)
        self.dtheta[1:] = np.degrees(np.arccos(cos))
        # the frames' candidates, laid end to end
        cand, at = rec.candidates_of(cbro)
        self.cand = cand
        self.at = at  # position of each candidate's frame among the CBRO frames
        o = origins(rec)
        walk = cand["origin"] <= o["engine top"]
        self.walk = cand[walk]
        self.walk_at = at[walk]
        self.walk_cls = classify(rec, self.walk)
        ids, self.walk_id = np.unique(self.walk["object"], return_inverse=True)
        self.walk_objects = ids
        self.N = len(ids)

    def class_matrix(self):
        """C[id, k]: each walk entry's class per frame (0 absent). A second decision in one frame keeps the last."""
        C = np.zeros((self.N, self.F), dtype=np.uint8)
        C[self.walk_id, self.walk_at] = self.walk_cls
        return C

    def per_frame(self, mask=None, weights=None):
        """A per-frame count of the walk candidates selected by mask."""
        at = self.walk_at if mask is None else self.walk_at[mask]
        return np.bincount(at, weights=weights, minlength=self.F)

    def cand_per_frame(self, mask):
        return np.bincount(self.at[mask], minlength=self.F)


def pct(a, qs=(50, 90, 99)):
    return " ".join(f"p{q} {np.percentile(a, q):.1f}" for q in qs)


def lstsq(y, *xs):
    X = np.column_stack([np.ones(len(y))] + list(xs))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    return coef, 1.0 - ss_res / ss_tot


# ---- commands ----------------------------------------------------------------------------------------------------
def cmd_inputs(r, args):
    print(f"{r.rec.path.name}: {r.F} CBRO frames")
    print(f"translation |dp| per frame (units): mean {r.dp.mean():.2f}, {pct(r.dp)}, max {r.dp.max():.1f}; total {r.dp.sum():.0f}")
    print(f"rotation |dtheta| per frame (deg): mean {r.dtheta.mean():.3f}, {pct(r.dtheta)}, max {r.dtheta.max():.2f}; total {r.dtheta.sum():.0f}")
    still = (r.dp < 0.5) & (r.dtheta < 0.05)
    turning = r.dtheta >= 0.5
    print(f"frames: still {int(still.sum())}, moving only {int(((r.dp >= 0.5) & (r.dtheta < 0.05)).sum())}, turning >= 0.5 deg {int(turning.sum())}")
    z = r.frames["zoom"]
    print(f"zoom (frustum half-extents): x {z[:, 0].min():.4f}..{z[:, 0].max():.4f}, y {z[:, 1].min():.4f}..{z[:, 1].max():.4f}")
    print(f"async map valid {int(r.frames['asyncValid'].sum())} of {r.F}; settled {int(r.frames['settled'].sum())}")


def work_arrays(r):
    rec = r.rec
    o = origins(rec)
    c, at = r.cand, r.at
    flags = c["flags"].astype(np.int64)
    judged = (flags & (1 << 8)) != 0
    src = c["source"]
    snames = {n: i for i, n in enumerate(rec.enums["source"])}
    w = {}
    leased = r.walk_cls == LEASED
    w["walk entries"] = r.per_frame()
    w["walk touched"] = r.per_frame(~leased)
    w["walk leased"] = r.per_frame(leased)
    w["walk cheap skip"] = r.per_frame(r.walk_cls == CHEAP)
    w["walk offered"] = r.per_frame((r.walk["origin"] == o["walk top"]) | (r.walk["origin"] == o["engine top"]))
    w["walk filed"] = r.per_frame(r.walk_cls == FILED)
    w["walk sun only"] = r.per_frame(r.walk_cls == DROP)
    w["children"] = r.cand_per_frame(c["origin"] == o["child"])
    w["children kept"] = r.cand_per_frame((c["origin"] == o["child"]) & ((flags & (1 << 9)) == 0))
    w["registered"] = r.cand_per_frame(c["origin"] == o["registered"])
    w["refused"] = r.cand_per_frame(c["origin"] == o["refused"])
    w["judged fresh"] = r.cand_per_frame(judged & (src == snames["fresh"]))
    w["judged reused"] = r.cand_per_frame(judged & (src == snames["reused"]))
    w["judged async map"] = r.cand_per_frame(judged & (src == snames["async map"]))
    w["nodes pruned"] = r.cand_per_frame(c["origin"] == o["node pruned"])
    w["nodes walked"] = r.cand_per_frame(c["origin"] == o["node walked"])
    return w


def cmd_work(r, args):
    w = work_arrays(r)
    print(f"{'per frame':<18} {'mean':>8} {'p10':>8} {'p50':>8} {'p90':>8} {'max':>8} | fit on the input: a + b|dp| + c|dtheta| (R^2)")
    for name, v in w.items():
        coef, r2 = lstsq(v.astype(np.float64), r.dp, r.dtheta)
        print(f"{name:<18} {v.mean():>8.0f} {np.percentile(v, 10):>8.0f} {np.percentile(v, 50):>8.0f} {np.percentile(v, 90):>8.0f} {v.max():>8.0f} | {coef[0]:8.0f} {coef[1]:+7.1f}/unit {coef[2]:+7.1f}/deg ({r2:.2f})")


def cmd_states(r, args):
    C = r.class_matrix()
    NC = len(CLASS_NAMES)
    present = C != ABSENT
    print(f"walk entries: {r.N} distinct objects over {r.F} frames; present per frame mean {present.sum(0).mean():.0f}")
    counts = np.stack([(C == k).sum(0) for k in range(NC)])
    print("class counts per frame (mean, min..max):")
    for k in range(1, NC):
        if counts[k].max():
            print(f"  {CLASS_NAMES[k]:<11} {counts[k].mean():8.0f}  {counts[k].min():6d}..{counts[k].max():6d}")
    # transitions between consecutive frames
    pair = C[:, :-1].astype(np.int64) * NC + C[:, 1:]
    T = np.bincount(pair.ravel(), minlength=NC * NC).reshape(NC, NC).astype(np.float64)
    steps = r.F - 1
    print("\ntransitions per frame (rows: class at k-1, columns: class at k; absent->absent left out):")
    shown = [k for k in range(NC) if T[k].sum() + T[:, k].sum() > 0]
    print(f"{'':<11} " + " ".join(f"{CLASS_NAMES[k][:9]:>9}" for k in shown))
    for a in shown:
        row = " ".join(f"{(T[a, b] / steps if not (a == 0 and b == 0) else 0):>9.1f}" for b in shown)
        print(f"{CLASS_NAMES[a]:<11} {row}")
    # macro (main-view) classes
    M = MACRO[C]
    changed = (M[:, 1:] != M[:, :-1]) & (M[:, 1:] != 0) & (M[:, :-1] != 0)
    appear = (M[:, :-1] == 0) & (M[:, 1:] != 0)
    vanish = (M[:, :-1] != 0) & (M[:, 1:] == 0)
    innov = np.zeros(r.F)
    innov[1:] = changed.sum(0)
    print(f"\ninnovation (main-view class changed between consecutive frames, both present): mean {innov[1:].mean():.1f}/frame, {pct(innov[1:])}, max {innov.max():.0f}")
    print(f"entries appearing (absent -> present) {appear.sum(0).mean():.1f}/frame, vanishing {vanish.sum(0).mean():.1f}/frame")
    MT = np.bincount((M[:, :-1].astype(np.int64) * 5 + M[:, 1:]).ravel(), minlength=25).reshape(5, 5) / steps
    print("main-view class transitions per frame:")
    print(f"{'':<9} " + " ".join(f"{n:>9}" for n in MACRO_NAMES))
    for a in range(5):
        print(f"{MACRO_NAMES[a]:<9} " + " ".join(f"{(MT[a, b] if not (a == 0 and b == 0) else 0):>9.1f}" for b in range(5)))
    # touched work that re-confirmed an unchanged class (beyond filing what is drawn)
    touched = present & (C != LEASED)
    same = np.zeros_like(touched)
    same[:, 1:] = (M[:, 1:] == M[:, :-1]) & (M[:, :-1] != 0)
    redundant = touched & same & (M != 1) & (M != 2)  # (drawn and sun-only entries are filed every frame: not redundant)
    print(f"\ntouched entries per frame {touched.sum(0).mean():.0f}: filed (drawn or sun only) {(touched & ((M == 1) | (M == 2))).sum(0).mean():.0f}, "
          f"re-confirmed an unchanged hidden/out class {redundant.sum(0).mean():.0f}, changed or new {(touched & ~same & (M != 1) & (M != 2)).sum(0).mean():.0f}")
    # persistence: run lengths of the macro class among present frames
    print("\npersistence (runs of one main-view class over consecutive frames, by class; frames):")
    for m in (1, 2, 3, 4):
        runs = run_lengths(M, m)
        if len(runs):
            w = runs.astype(np.float64)
            print(f"  {MACRO_NAMES[m]:<9} runs {len(runs):7d}  mean {runs.mean():6.1f}  frame-weighted mean {float((w * w).sum() / w.sum()):6.1f}  {pct(runs)}  (runs cut by the recording's ends included)")
    coef, r2 = lstsq(innov[1:], r.dp[1:], r.dtheta[1:])
    print(f"\nfit innovation = {coef[0]:.1f} + {coef[1]:.2f}|dp| + {coef[2]:.1f}|dtheta|  (R^2 {r2:.2f})")


def run_lengths(M, m):
    """Lengths of the runs of class m along each row of M."""
    x = (M == m).astype(np.int8)
    pad = np.zeros((x.shape[0], 1), dtype=np.int8)
    d = np.diff(np.concatenate([pad, x, pad], axis=1), axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    # argwhere returns row-major order: starts and ends pair up row by row
    return (ends[:, 1] - starts[:, 1]).astype(np.int64)


def outside_distance(r, k, b):
    """How far outside the current view each sphere b (n x 4) lies in frame k, in world units, by CBRO's own test
    (SphereOutsideView: the depth frame's view space and the current view's rectangle in its NDC), without the
    verdict's margins. Negative: inside (by that much)."""
    f = r.frames[k]
    o = f["origin"].astype(np.float64)
    d = b[:, :3].astype(np.float64) - o
    z = d @ f["viewDir"].astype(np.float64)
    x = d @ f["viewRight"].astype(np.float64)
    y = d @ f["viewUp"].astype(np.float64)
    rad = b[:, 3].astype(np.float64)
    sx, sy = float(f["scaleX"]), float(f["scaleY"])
    v = f["view"].astype(np.float64)
    out = -(z + rad)  # behind the eye plane by this much
    for scale, coord, edge, upper in ((sx, x, v[0], False), (sx, x, v[1], True), (sy, y, v[2], False), (sy, y, v[3], True)):
        n = math.sqrt(scale * scale + edge * edge)
        side = (scale * coord - edge * z) / n  # >= 0 inside a lower edge, <= 0 inside an upper edge
        dist = (side if upper else -side) - rad
        out = np.maximum(out, dist)
    return out, np.sqrt((d * d).sum(1))


def trace_turn(r, objects, frame_pos, refresh):
    """TraceFeed::Turn: an object's turn for the trace feed (one frame in refresh, spread by address)."""
    h = (((objects.astype(np.uint64) >> np.uint64(4)) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)) & np.uint64(0xFFFFFFFF)
    clock = r.frames["clock"].astype(np.uint64)[frame_pos]
    return ((h + clock) & np.uint64(refresh - 1)) == 0


def cmd_certify(r, args):
    safety = float(args[args.index("--safety") + 1]) if "--safety" in args else 64.0
    max_lease = int(args[args.index("--max-lease") + 1]) if "--max-lease" in args else 16
    refresh = int(args[args.index("--trace-frames") + 1]) if "--trace-frames" in args else 64
    settle = int(args[args.index("--settle") + 1]) if "--settle" in args else 2
    if not r.frames["viewSpace"].all():
        print(f"warning: {int((r.frames['viewSpace'] == 0).sum())} frames without the view-space form: their entries count as touched")
    out_cls = (r.walk_cls == CHEAP) | (r.walk_cls == LEASED) | (r.walk_cls == OUTSIDE) | (r.walk_cls == UNNEEDED)
    turn_all = trace_turn(r, r.walk["object"], r.walk_at, refresh)
    T = np.cumsum(r.dp)
    PHI = np.cumsum(np.radians(r.dtheta))
    N = r.N
    issued_at = np.full(N, -1, dtype=np.int64)
    issued_T = np.zeros(N)
    issued_P = np.zeros(N)
    budget = np.zeros(N)
    reach = np.zeros(N)
    issued_b = np.zeros((N, 4), dtype=np.float32)
    last_b = np.full((N, 4), np.nan, dtype=np.float32)   # the bound at the scheme's last touch
    still = np.zeros(N, dtype=np.int64)                  # consecutive touches with an unchanged bound
    reasons = Counter()
    misses = Counter()
    stale_valid = 0
    needed = np.zeros(r.F, dtype=np.int64)
    lease_lengths = []
    order = np.argsort(r.walk_at, kind="stable")
    bounds = np.stack([r.walk["bound"][:, i] for i in range(4)], axis=1).astype(np.float32)
    starts = np.searchsorted(r.walk_at[order], np.arange(r.F + 1))
    for k in range(r.F):
        rows = order[starts[k]:starts[k + 1]]
        ids = r.walk_id[rows]
        cls = r.walk_cls[rows]
        b = bounds[rows]
        turn = turn_all[rows]
        has = issued_at[ids] >= 0
        motion_ok = T[k] - issued_T[ids] + (PHI[k] - issued_P[ids]) * reach[ids] < budget[ids]
        age_ok = k - issued_at[ids] < max_lease
        valid = has & motion_ok & age_ok
        moved_under = valid & np.any(b != issued_b[ids], axis=1)
        stale_valid += int(moved_under.sum())
        bad = valid & ~out_cls[rows] & ~turn
        for c, mv in zip(cls[bad], moved_under[bad]):
            misses[(CLASS_NAMES[c], "moved under it" if mv else "unmoved")] += 1
        touch = ~valid | turn
        needed[k] = int(touch.sum())
        reasons["certificate valid: untouched"] += int((~touch).sum())
        reasons["trace turn"] += int((valid & turn).sum())
        reasons["no certificate"] += int((~has).sum())
        reasons["expired: motion"] += int((has & ~motion_ok).sum())
        reasons["expired: age"] += int((has & motion_ok & ~age_ok).sum())
        # the touch: the scheme sees the bound now
        tid = ids[touch]
        tb = b[touch]
        same = np.all(tb == last_b[tid], axis=1)
        still[tid] = np.where(same, still[tid] + 1, 0)
        last_b[tid] = tb
        eligible = touch.copy()
        eligible[touch] = out_cls[rows][touch] & (still[tid] >= settle - 1)
        issued_at[ids[touch & ~eligible]] = -1
        if eligible.any() and r.frames["viewSpace"][k]:
            sel = rows[eligible]
            dist, eye_dist = outside_distance(r, k, bounds[sel])
            ok = dist > safety
            nid = ids[eligible][ok]
            ended = issued_at[nid] >= 0
            lease_lengths.extend((k - issued_at[nid][ended]).tolist())
            issued_at[nid] = k
            issued_T[nid] = T[k]
            issued_P[nid] = PHI[k]
            budget[nid] = dist[ok] - safety
            reach[nid] = eye_dist[ok] + bounds[sel][ok, 3]
            issued_b[nid] = bounds[sel][ok]
            issued_at[ids[eligible][~ok]] = -1
    actual = r.per_frame(r.walk_cls != LEASED)
    filed = r.per_frame((r.walk_cls == FILED) | (r.walk_cls == DROP) | (r.walk_cls == MARKER))
    print(f"certificates: safety {safety:.0f} units, lease at most {max_lease} frames, trace turn 1 in {refresh}, settled after {settle} equal bounds")
    print(f"walk entries touched per frame: recording {actual.mean():.0f} (p90 {np.percentile(actual, 90):.0f}) | certificate scheme {needed.mean():.0f} (p90 {np.percentile(needed, 90):.0f}); filed every frame anyway {filed.mean():.0f}")
    total = sum(reasons.values())
    print("entry-frames: " + ", ".join(f"{k} {v / r.F:.0f}/frame" for k, v in reasons.items()))
    print(f"certificates valid on an entry whose bound had changed since: {stale_valid / r.F:.2f}/frame")
    nm = sum(misses.values())
    print(f"misses (a valid certificate on an entry the recording drew, dropped for the sun or judged hidden, off its trace turn): {nm} entry-frames ({nm / r.F:.2f}/frame)" + (": " + ", ".join(f"{a} ({b}) {v}" for (a, b), v in misses.most_common()) if nm else ""))
    if lease_lengths:
        ll = np.array(lease_lengths)
        print(f"certificates renewed after (frames): mean {ll.mean():.1f}, {pct(ll)}")


def cmd_drops(r, args):
    rec = r.rec
    o = origins(rec)
    p = paths(rec)
    c, at = r.cand, r.at
    child = c["origin"] == o["child"]
    w = {
        "top entries dropped for the sun only": r.per_frame(r.walk_cls == DROP),
        "children: main view drop": r.cand_per_frame(child & (c["path"] == p["main view drop"])),
        "children: parent dropped": r.cand_per_frame(child & (c["path"] == p["parent dropped"])),
        "registrations refused": r.cand_per_frame(c["origin"] == o["refused"]),
        "registrations kept": r.cand_per_frame(c["origin"] == o["registered"]),
    }
    for name, v in w.items():
        print(f"{name:<40} {v.mean():8.0f}/frame  ({pct(v)})")
    # which of the refused registrations belong to a dropped top entry's subtree (by the objects table's parent chain)
    refused = c[c["origin"] == o["refused"]]
    rows = rec.object_rows(refused["object"])
    parents = np.where(rows >= 0, rec.objects["parent"][np.maximum(rows, 0)], 0)
    drop_tops = set(int(x) for x in r.walk["object"][r.walk_cls == DROP])
    in_drop = np.array([int(x) in drop_tops for x in parents], dtype=bool)
    print(f"refused registrations whose parent is a dropped top entry: {in_drop.mean() * 100:.0f}%")


def cmd_parents(r, args):
    rec = r.rec
    o = origins(rec)
    c = r.cand[r.cand["origin"] == o["child"]]
    rows = rec.object_rows(c["object"])
    par = np.where(rows >= 0, rec.objects["parent"][np.maximum(rows, 0)], 0)
    prow = rec.object_rows(par)
    known = prow >= 0
    rt = np.where(known, rec.objects["rtti"][np.maximum(prow, 0)].astype(np.int64), -1)
    nm = np.where(known, rec.objects["name"][np.maximum(prow, 0)].astype(np.int64), -1)
    fade = np.where(known, rec.objects["isFadeNode"][np.maximum(prow, 0)], 0)
    F = r.F
    print(f"children per frame {len(c) / F:.0f}; parent recorded {float(known.mean()) * 100:.0f}% (a parent that is a sub-node of a filed node is not), of them a fade node {float(fade[known].mean()) * 100:.0f}%")
    tk, tc = np.unique(rt, return_counts=True)
    print("by parent type: " + ", ".join(f"{rec.string(int(t)) if t >= 0 else '(not recorded)'} {n / F:.0f}" for t, n in sorted(zip(tk, tc), key=lambda kv: -kv[1])[:12]))
    keys, counts = np.unique(rt * (1 << 32) + np.where(nm >= 0, nm, 0), return_counts=True)
    print("commonest recorded parents (type 'name'): per frame")
    for k, n in sorted(zip(keys, counts), key=lambda kv: -kv[1])[:25]:
        t = int(k) >> 32
        if t < 0:
            continue
        print(f"  {n / F:8.1f}  {rec.string(t)} '{rec.string(int(k) & 0xFFFFFFFF)}'")
    # the children's own types
    crow = rows[rows >= 0]
    ck, cc = np.unique(rec.objects["rtti"][crow], return_counts=True)
    print("children by their own type: " + ", ".join(f"{rec.string(int(t))} {n / F:.0f}" for t, n in sorted(zip(ck, cc), key=lambda kv: -kv[1])[:12]))


def cmd_children(r, args):
    rec = r.rec
    o = origins(rec)
    p = paths(rec)
    c, at = r.cand, r.at
    is_child = c["origin"] == o["child"]
    ch = c[is_child]
    ch_at = at[is_child]
    rows = rec.object_rows(ch["object"])
    par = np.where(rows >= 0, rec.objects["parent"][np.maximum(rows, 0)], 0).astype(np.uint64)
    # each frame's top decision by object
    key_top = (r.walk_at.astype(np.uint64) << np.uint64(48)) ^ r.walk["object"].astype(np.uint64)
    order = np.argsort(key_top)
    sk = key_top[order]
    key_ch = (ch_at.astype(np.uint64) << np.uint64(48)) ^ par
    pos = np.minimum(np.searchsorted(sk, key_ch), len(sk) - 1)
    hit = sk[pos] == key_ch
    top_cls = np.where(hit, r.walk_cls[order][pos], 255)
    left = (ch["flags"].astype(np.int64) & (1 << 9)) != 0
    F = r.F
    print(f"children per frame {len(ch) / F:.0f}; with their parent decided at the top level that frame {hit.mean() * 100:.0f}%")
    tab = Counter()
    for tcls, lft, pth in zip(top_cls[hit], left[hit], ch["path"][hit]):
        tab[(CLASS_NAMES[int(tcls)], "left out" if lft else "kept", rec.name("path", int(pth)))] += 1
    print(f"{'parent class':<12} {'child':<9} {'path':<36} per frame")
    for (a, b, pth), n in tab.most_common(20):
        print(f"{a:<12} {b:<9} {pth:<36} {n / F:8.1f}")
    filed_parent = hit & (top_cls == FILED)
    agree = filed_parent & ~left
    print(f"children of filed parents: {filed_parent.sum() / F:.0f}/frame; kept {agree.sum() / F:.0f}, left out {(filed_parent & left).sum() / F:.0f}")


def cmd_moved(r, args):
    rec = r.rec
    moved = (r.walk["flags"].astype(np.int64) & (1 << 6)) != 0
    print(f"walk entries with a changed bound per frame {moved.sum() / r.F:.1f} (a new slot counts); distinct objects {len(np.unique(r.walk['object'][moved]))}")
    rows = rec.object_rows(r.walk["object"][moved])
    ok = rows >= 0
    keys = (rec.objects["rtti"][rows[ok]].astype(np.int64) << 32) | rec.objects["name"][rows[ok]].astype(np.int64)
    u, n = np.unique(keys, return_counts=True)
    for k, c in sorted(zip(u, n), key=lambda kv: -kv[1])[:20]:
        print(f"  {c / r.F:7.1f}/frame  {rec.string(int(k) >> 32)} '{rec.string(int(k) & 0xFFFFFFFF)}'")
    cu, cn = np.unique(r.walk_cls[moved], return_counts=True)
    print("by class: " + ", ".join(f"{CLASS_NAMES[int(a)]} {b / r.F:.1f}" for a, b in zip(cu, cn)))
    o, b, at = r.walk["object"][moved], r.walk["bound"][moved], r.walk_at[moved]
    order = np.lexsort((at, o))
    o, b = o[order], b[order]
    same = o[1:] == o[:-1]
    if same.any():
        d = np.linalg.norm((b[1:, :3] - b[:-1, :3]).astype(np.float64), axis=1)[same]
        print(f"shift between an object's consecutive changed records: centre {pct(d)}, max {d.max():.1f}")


def cmd_feed(r, args):
    """The filed and sun-only classes split by why: the trace's turn, a move, a force-visible filing, or neither."""
    refresh = int(args[args.index("--trace-frames") + 1]) if "--trace-frames" in args else 64
    fl = r.walk["flags"].astype(np.int64)
    turn = trace_turn(r, r.walk["object"], r.walk_at, refresh)
    moved = (fl & (1 << 6)) != 0
    forced = (fl & 1) != 0
    for cls in (FILED, DROP):
        m = r.walk_cls == cls
        tab = Counter()
        for t_, mv, fo in zip(turn[m], moved[m], forced[m]):
            tab[("trace turn" if t_ else "-", "moved" if mv else "-", "forced" if fo else "-")] += 1
        print(f"{CLASS_NAMES[cls]} {m.sum() / r.F:.0f}/frame: " + ", ".join(f"{'/'.join(k)} {v / r.F:.1f}" for k, v in tab.most_common()))
    real = (r.walk_cls == DROP) & ~turn & ~moved & ~forced
    print(f"dropped for the sun only, none of those reasons (its shadow needed): {real.sum() / r.F:.0f}/frame")


COMMANDS = {"inputs": cmd_inputs, "work": cmd_work, "states": cmd_states, "certify": cmd_certify, "drops": cmd_drops,
            "children": cmd_children, "parents": cmd_parents, "moved": cmd_moved, "feed": cmd_feed}


def main(argv):
    if len(argv) < 1 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    command = argv[1] if len(argv) > 1 else "states"
    if command not in COMMANDS:
        print(f"unknown command {command}\n{__doc__}")
        return 2
    r = Replay(argv[0])
    return COMMANDS[command](r, argv[2:]) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
