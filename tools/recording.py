"""Reads a CBRO recording (Core/Recorder, bRecorder; Data/CBRO_Recordings/*.cbrorec) without the game.

usage: python -I tools/recording.py <file.cbrorec> [command] [options]
  summary               frames by mode, candidates by origin per frame, memory (default)
  verdicts              histogram of outcome x verdict x sun x source over the CBRO frames' judged candidates
  origins               candidates by origin and site, kept against left out, by path
  noverdict             entries decided without a verdict (walk skips, forced, inherited, markers): by origin and reason,
                        the commonest object types and names
  diff [--radius R] [--angle A] [--other FILE]
                        frame set differences: each previs frame is matched with the nearest CBRO frame (camera within
                        R units, default 48, and A degrees of yaw, default 6) and the main-view registrations compared:
                        objects previs drew that CBRO never registered in the matched frame (with CBRO's decision on
                        them that frame, when it made one), and the reverse. --other compares against a second
                        recording's frames instead (objects matched by reference id and name, not address).
  frame N               one frame's header and its candidates' counts
  objects REGEX         objects whose type or name matches, with the frames and decisions that name them
  check                 integrity (sizes, offsets); exit code 1 on a problem (the round-trip test uses it)

Everything runs as numpy array work: no command loops over candidates in Python (15M of them in an outdoor minute).
Each candidate carries its frame's index (Recording.frame_of); objects are found by address through one sorted table;
distinct combinations of fields come from np.unique over a packed copy (group_counts).
"""
import json
import math
import re
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # python -I leaves the user site-packages out: look there before giving up
    import site

    sys.path.append(site.getusersitepackages())
    try:
        import numpy as np
    except ModuleNotFoundError:
        sys.exit("recording.py needs numpy (pip install numpy)")

HEADER = struct.Struct("<8sIIIIQIIQQIIIIII")  # FileHeader, 80 bytes
assert HEADER.size == 80


def align8(n):
    return (n + 7) & ~7


def dtype_of(fields):
    out = []
    for name, kind, count in fields:
        np_type = "<" + kind if kind[0] in "uif" and kind[1:].isdigit() else kind
        out.append((name, np_type, (count,)) if count > 1 else (name, np_type))
    return np.dtype(out)


# ---- array helpers ---------------------------------------------------------------------------------------------------
def ranges(starts, counts):
    """The indices of the ranges [start, start + count) laid end to end, in order (one np.arange, no loop)."""
    starts = np.asarray(starts, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    before = np.cumsum(counts) - counts
    return np.arange(total, dtype=np.int64) - np.repeat(before, counts) + np.repeat(starts, counts)


def packed(table, fields):
    """A contiguous copy of the scalar fields, for np.unique (distinct combinations, sorted field by field)."""
    arr = np.empty(len(table), dtype=[(f, table.dtype[f]) for f in fields])
    for f in fields:
        arr[f] = table[f]
    return arr


def group_counts(table, fields):
    """The distinct combinations of `fields` in `table` and how often each occurs."""
    return np.unique(packed(table, fields), return_counts=True)


def most_common(keys, counts, n=None):
    """(key, count) pairs, largest count first (ties by key order)."""
    order = np.argsort(-counts, kind="stable")
    if n is not None:
        order = order[:n]
    return [(keys[i], int(counts[i])) for i in order]


class Recording:
    def __init__(self, path):
        self.path = Path(path)
        data = self.path.read_bytes()
        self.bytes = len(data)
        (magic, version, json_bytes, frame_count, object_count, candidate_count, lamp_count, hiz_blobs, hiz_bytes,
         string_bytes, frame_bytes, candidate_bytes, lamp_bytes, object_bytes, blob_bytes, _reserved) = HEADER.unpack_from(data, 0)
        if magic != b"CBROREC\0":
            raise ValueError(f"{path}: not a CBRO recording (magic {magic!r})")
        if version != 1:
            raise ValueError(f"{path}: format version {version}, this reader knows 1")
        pos = HEADER.size
        self.meta = json.loads(data[pos:pos + json_bytes].decode("utf-8"))
        pos = align8(pos + json_bytes)
        structs = self.meta["structs"]
        self.frame_dtype = dtype_of(structs["frame"])
        self.candidate_dtype = dtype_of(structs["candidate"])
        self.lamp_dtype = dtype_of(structs["lamp"])
        self.object_dtype = dtype_of(structs["object"])
        for name, dtype, size in (("frame", self.frame_dtype, frame_bytes), ("candidate", self.candidate_dtype, candidate_bytes),
                                  ("lamp", self.lamp_dtype, lamp_bytes), ("object", self.object_dtype, object_bytes)):
            if dtype.itemsize != size:
                raise ValueError(f"{path}: the {name} record is {size} bytes in the file, {dtype.itemsize} by its field list")

        def section(dtype, count):
            nonlocal pos
            n = dtype.itemsize * count
            if pos + n > len(data):
                raise ValueError(f"{path}: truncated at {pos} (+{n})")
            arr = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
            pos = align8(pos + n)
            return arr

        self.frames = section(self.frame_dtype, frame_count)
        self.candidates = section(self.candidate_dtype, candidate_count)
        self.lamps = section(self.lamp_dtype, lamp_count)
        self.hiz_level = self.meta["hiz"]["level"]
        self.hiz_width = self.meta["hiz"]["width"]
        self.hiz_height = self.meta["hiz"]["height"]
        self.blob_bytes = blob_bytes
        self.hiz = None
        if hiz_blobs and blob_bytes:
            floats = blob_bytes // 4
            if pos + hiz_bytes > len(data) or floats != 2 * self.hiz_width * self.hiz_height:
                raise ValueError(f"{path}: hi-z section inconsistent ({hiz_blobs} blobs x {blob_bytes} bytes, {self.hiz_width}x{self.hiz_height})")
            self.hiz = np.frombuffer(data, dtype="<f4", count=hiz_blobs * floats, offset=pos).reshape(hiz_blobs, 2, self.hiz_height, self.hiz_width)
        pos = align8(pos + hiz_bytes)
        self.objects = section(self.object_dtype, object_count)
        self.strings = data[pos:pos + string_bytes]
        pos += string_bytes
        if pos > len(data):
            raise ValueError(f"{path}: truncated strings")
        self.enums = self.meta["enums"]
        self.object_index = {int(o): i for i, o in enumerate(self.objects["object"])}
        # each candidate's frame (frames own runs of the candidates section; a candidate no frame owns keeps -1)
        starts = self.frames["candidateOffset"].astype(np.int64)
        counts = self.frames["candidateCount"].astype(np.int64)
        owned = ranges(starts, counts)
        owner = np.repeat(np.arange(len(self.frames), dtype=np.int32), counts)
        inside = owned < len(self.candidates)
        self.frame_of = np.full(len(self.candidates), -1, dtype=np.int32)
        self.frame_of[owned[inside]] = owner[inside]
        # objects by address: one sorted table, looked up with searchsorted
        self._object_order = np.argsort(self.objects["object"], kind="stable")
        self._object_sorted = self.objects["object"][self._object_order]
        self._string_cache = {}

    # ---- helpers ---------------------------------------------------------------------------------------------------
    def string(self, offset):
        hit = self._string_cache.get(offset)
        if hit is None:
            end = self.strings.find(b"\0", offset)
            hit = self._string_cache[offset] = self.strings[offset:end if end >= 0 else None].decode("utf-8", errors="replace")
        return hit

    def strings_at(self, offsets):
        """The strings at an array of offsets (each distinct string decoded once)."""
        unique, inverse = np.unique(np.asarray(offsets, dtype=np.int64), return_inverse=True)
        table = np.array([self.string(int(o)) for o in unique] + [""], dtype=object)  # (+"" keeps an empty table indexable)
        return table[inverse]

    def name(self, enum, value):
        names = self.enums[enum]
        return names[value] if 0 <= value < len(names) else f"{enum}?{value}"

    def flag_names(self, flags):
        return [n for i, n in enumerate(self.enums["flags"]) if flags & (1 << i)]

    def flag_mask(self, names):
        return sum(1 << i for i, n in enumerate(self.enums["flags"]) if n in names)

    def frame_candidates(self, i):
        f = self.frames[i]
        a, n = int(f["candidateOffset"]), int(f["candidateCount"])
        return self.candidates[a:a + n]

    def candidates_of(self, frame_indices):
        """The candidates of the frames given, laid end to end, with the position of each one's frame in the list."""
        frames = self.frames[np.asarray(frame_indices, dtype=np.int64)]
        counts = frames["candidateCount"].astype(np.int64)
        idx = ranges(frames["candidateOffset"], counts)
        at = np.repeat(np.arange(len(frames), dtype=np.int64), counts)
        inside = idx < len(self.candidates)
        return self.candidates[idx[inside]], at[inside]

    def object_rows(self, addresses):
        """The objects-section row of each address (-1: not recorded)."""
        addresses = np.asarray(addresses, dtype=np.uint64)
        if len(self._object_sorted) == 0:
            return np.full(len(addresses), -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self._object_sorted, addresses), len(self._object_sorted) - 1)
        rows = self._object_order[pos].astype(np.int64)
        rows[self._object_sorted[pos] != addresses] = -1
        return rows

    def describe(self, address):
        i = self.object_index.get(int(address))
        if i is None:
            return "<unknown object>"
        o = self.objects[i]
        ref = f" ref {int(o['refID']):08X}" if o["refID"] else ""
        base = f" base {int(o['baseID']):08X}" if o["baseID"] else ""
        name = self.string(int(o["name"]))
        return f"{self.string(int(o['rtti']))} '{name}'{ref}{base}"

    def object_key(self, address):
        """A key that survives a new game session: reference id + base id + name (the address does not)."""
        i = self.object_index.get(int(address))
        if i is None:
            return ("?", int(address))
        o = self.objects[i]
        return (int(o["refID"]), int(o["baseID"]), self.string(int(o["name"])), self.string(int(o["rtti"])))

    def yaw(self, frame):
        r = frame["rotate"]
        # the WorldRoot camera's view direction is its rotation's second row (y forward) in this engine; the angle between two
        # frames' rows is what matters, so any consistent row will do
        return math.degrees(math.atan2(float(r[4]), float(r[3])))

    def yaws(self, frames):
        r = frames["rotate"].astype(np.float64).reshape(len(frames), -1)
        return np.degrees(np.arctan2(r[:, 4], r[:, 3]))


class KeySpace:
    """Small integer ids for objects, so frame sets compare as integer arrays. Within one recording an object's id is its
    row (addresses are unique in a session); across two, the key is (reference id, base id, name, type), which survives
    a new session. A candidate whose object the objects section lacks gets an id of its own by address."""

    def __init__(self, rec, other=None):
        self.keyed = other is not None and other is not rec
        self.ids = {}
        if not self.keyed:
            self.ids[id(rec)] = np.arange(len(rec.objects), dtype=np.int64)
            self.rec = rec
            self.known = len(rec.objects)
        else:
            lookup = {}
            for r in (rec, other):
                names = r.strings_at(r.objects["name"])
                rttis = r.strings_at(r.objects["rtti"])
                ids = np.empty(len(r.objects), dtype=np.int64)
                for i, key in enumerate(zip(r.objects["refID"].tolist(), r.objects["baseID"].tolist(), names.tolist(), rttis.tolist())):
                    ids[i] = lookup.setdefault(key, len(lookup))
                self.ids[id(r)] = ids
            self.keys = list(lookup)
            self.known = len(self.keys)
        self.unknown = {}
        self.addresses = []

    def of(self, rec, addresses):
        """The ids of the candidates' objects."""
        addresses = np.asarray(addresses, dtype=np.uint64)
        rows = rec.object_rows(addresses)
        out = np.empty(len(rows), dtype=np.int64)
        known = rows >= 0
        out[known] = self.ids[id(rec)][rows[known]]
        if not known.all():
            unique, inverse = np.unique(addresses[~known], return_inverse=True)
            ids = np.empty(len(unique), dtype=np.int64)
            for i, a in enumerate(unique.tolist()):
                if a not in self.unknown:
                    self.unknown[a] = self.known + len(self.addresses)
                    self.addresses.append(a)
                ids[i] = self.unknown[a]
            out[~known] = ids[inverse]
        return out

    def describe(self, key_id):
        key_id = int(key_id)
        if key_id >= self.known:
            return f"<unknown object> at {self.addresses[key_id - self.known]:X}"
        if not self.keyed:
            return self.rec.describe(int(self.rec.objects[key_id]["object"]))
        ref, base, name, rtti = self.keys[key_id]
        return f"{rtti} '{name}' ref {ref:08X} base {base:08X}"


# ---- commands --------------------------------------------------------------------------------------------------------
def cmd_summary(rec, args):
    print(f"{rec.path.name}: {rec.bytes / 1048576:.1f} MB, CBRO {rec.meta.get('cbro')}, recorded {rec.meta.get('recordedAt')}, frame limit {rec.meta.get('frameLimit')}")
    print(f"frames {len(rec.frames)} | candidates {len(rec.candidates)} | objects {len(rec.objects)} | lamps {len(rec.lamps)} | hi-z level {rec.hiz_level} {rec.hiz_width}x{rec.hiz_height}, blobs {0 if rec.hiz is None else len(rec.hiz)}")
    if len(rec.frames) == 0:
        return
    modes, mode_frames = np.unique(rec.frames["mode"], return_counts=True)
    print("by mode: " + ", ".join(f"{rec.name('mode', int(m))} {int(n)} frames" for m, n in zip(modes, mode_frames)))
    print(f"clock {int(rec.frames['clock'][0])}..{int(rec.frames['clock'][-1])}, render frames {int(rec.frames['renderFrame'][0])}..{int(rec.frames['renderFrame'][-1])}, settled {int(rec.frames['settled'].sum())}, culled {int(rec.frames['cull'].sum())}, async map used {int(rec.frames['asyncValid'].sum())}, overflowed candidates {int(rec.frames['overflow'].sum())}")
    print("candidates per frame by origin (mean over the mode's frames):")
    owned = rec.frame_of >= 0
    mode_of = rec.frames["mode"][rec.frame_of[owned]]
    origin_of = rec.candidates["origin"][owned]
    for m, frames in zip(modes, mode_frames):
        origins, counts = np.unique(origin_of[mode_of == m], return_counts=True)
        parts = ", ".join(f"{rec.name('origin', int(o))} {int(v) / int(frames):.0f}" for o, v in zip(origins, counts))
        print(f"  {rec.name('mode', int(m))}: {parts} | all {int(counts.sum()) / int(frames):.0f}")
    eyes = rec.frames["eye"]
    moved = np.linalg.norm(np.diff(eyes, axis=0), axis=1) if len(eyes) > 1 else np.zeros(0)
    if len(moved):
        print(f"camera: moved {moved.sum():.0f} units over the recording ({moved.mean():.2f} a frame, max {moved.max():.1f}); from ({eyes[0][0]:.0f},{eyes[0][1]:.0f},{eyes[0][2]:.0f}) to ({eyes[-1][0]:.0f},{eyes[-1][1]:.0f},{eyes[-1][2]:.0f})")
    states, counts = np.unique(rec.frames["sunState"], return_counts=True)
    print("sun: " + ", ".join(f"{rec.name('sunState', int(s))} {int(n)}" for s, n in zip(states, counts)) + f" | lamps per frame {len(rec.lamps) / len(rec.frames):.1f}")


def cbro_candidates(rec):
    """The candidates of the CBRO-mode frames (one mask through frame_of)."""
    owned = rec.frame_of >= 0
    mask = np.zeros(len(rec.candidates), dtype=bool)
    mask[owned] = rec.frames["mode"][rec.frame_of[owned]] != 0
    return rec.candidates[mask]


def cmd_verdicts(rec, args):
    c = cbro_candidates(rec)
    judged = c[(c["flags"] & (1 << 8)) != 0]
    print(f"CBRO frames: {int((rec.frames['mode'] != 0).sum())}, candidates {len(c)}, judged {len(judged)} ({100.0 * len(judged) / max(len(c), 1):.1f}%)")
    keys, counts = group_counts(judged, ("origin", "outcome", "verdict", "sun", "source"))
    print(f"{'origin':<14} {'outcome':<9} {'verdict':<9} {'sun':<14} {'source':<11} {'count':>9} {'share':>7}")
    for k, n in most_common(keys, counts, 60):
        print(f"{rec.name('origin', int(k['origin'])):<14} {rec.name('outcome', int(k['outcome'])):<9} {rec.name('verdict', int(k['verdict'])):<9} {rec.name('sun', int(k['sun'])):<14} {rec.name('source', int(k['source'])):<11} {n:>9} {100.0 * n / len(judged):>6.1f}%")
    streaks, counts = np.unique(judged["streak"][judged["outcome"] == 2], return_counts=True)
    if len(streaks):
        print("hidden streaks: " + ", ".join(f"{int(k)}: {int(v)}" for k, v in list(zip(streaks, counts))[:12]))


def cmd_origins(rec, args):
    c = cbro_candidates(rec)
    frames = max(int((rec.frames["mode"] != 0).sum()), 1)
    print(f"{'origin':<14} {'site':>5} {'all/frame':>10} {'left out':>9} {'kept':>7} | paths of the left out")
    group = (c["origin"].astype(np.int64) << 16) | c["site"].astype(np.int64)
    keys, inverse, sizes = np.unique(group, return_inverse=True, return_counts=True)
    left = (c["flags"] & (1 << 9)) != 0
    for g in np.argsort(-sizes, kind="stable"):
        rows = inverse == g
        paths, counts = np.unique(c["path"][rows & left], return_counts=True)
        detail = ", ".join(f"{rec.name('path', int(p))} {n / frames:.0f}" for p, n in most_common(paths, counts, 4))
        n_all, n_left = int(sizes[g]), int((rows & left).sum())
        print(f"{rec.name('origin', int(keys[g]) >> 16):<14} {int(keys[g]) & 0xFFFF:>5} {n_all / frames:>10.0f} {n_left / frames:>9.0f} {(n_all - n_left) / frames:>7.0f} | {detail}")


def cmd_noverdict(rec, args):
    c = cbro_candidates(rec)
    frames = max(int((rec.frames["mode"] != 0).sum()), 1)
    entries = c[(c["origin"] <= 4) & ((c["flags"] & (1 << 8)) == 0)]
    print(f"entries decided without a verdict: {len(entries)} ({len(entries) / frames:.0f} a frame) of {len(c)} candidates")
    hidden = rec.flag_mask(("left out", "group 1", "judged"))
    reason = (entries["origin"].astype(np.int64) << 32) | (entries["flags"].astype(np.int64) & ~hidden)
    keys, counts = np.unique(reason, return_counts=True)
    print(f"{'origin':<14} {'per frame':>10}  reason")
    for k, n in most_common(keys, counts, 30):
        why = ", ".join(rec.flag_names(int(k) & 0xFFFFFFFF)) or "(no flag)"
        print(f"{rec.name('origin', int(k) >> 32):<14} {n / frames:>10.0f}  {why}")
    rows = rec.object_rows(entries["object"][entries["origin"] != 0])  # the walk's cheap skips are by design: the rest is what to look at
    rows = rows[rows >= 0]
    if len(rows):
        objects = rec.objects[rows]
        rttis, counts = np.unique(objects["rtti"], return_counts=True)
        print("forced, inherited or marker-filed entries by type: " + ", ".join(f"{rec.string(int(t))} {n / frames:.0f}" for t, n in most_common(rttis, counts, 12)))
        print("... commonest objects:")
        names, counts = np.unique((objects["rtti"].astype(np.int64) << 32) | objects["name"].astype(np.int64), return_counts=True)
        for k, n in most_common(names, counts, 20):
            print(f"  {n / frames:>7.1f}/frame  {rec.string(int(k) >> 32)} '{rec.string(int(k) & 0xFFFFFFFF)}'")


def match_frames(rec, other, radius, angle):
    """For each previs frame of `rec`, the nearest culled CBRO frame of `other`: arrays of (previs index, CBRO index,
    distance, turn), one row per previs frame that has a match (one distance matrix, no loop)."""
    previs = np.nonzero(rec.frames["mode"] == 0)[0]
    cbro = np.nonzero((other.frames["mode"] != 0) & (other.frames["cull"] != 0))[0]
    empty = (np.zeros(0, dtype=np.int64),) * 2 + (np.zeros(0),) * 2
    if len(previs) == 0 or len(cbro) == 0:
        return empty
    eyes_p = rec.frames["eye"][previs].astype(np.float64)
    eyes_c = other.frames["eye"][cbro].astype(np.float64)
    d = np.linalg.norm(eyes_p[:, None, :] - eyes_c[None, :, :], axis=2)
    turn = np.abs((other.yaws(other.frames[cbro])[None, :] - rec.yaws(rec.frames[previs])[:, None] + 180.0) % 360.0 - 180.0)
    score = np.where((d <= radius) & (turn <= angle), d + turn * radius / angle, np.inf)
    j = np.argmin(score, axis=1)
    ok = np.isfinite(score[np.arange(len(previs)), j])
    i = np.nonzero(ok)[0]
    return previs[i], cbro[j[i]], d[i, j[i]], turn[i, j[i]]


def cmd_diff(rec, args):
    radius, angle, other_path = 48.0, 6.0, None
    it = iter(args)
    for a in it:
        if a == "--radius":
            radius = float(next(it))
        elif a == "--angle":
            angle = float(next(it))
        elif a == "--other":
            other_path = next(it)
    other = Recording(other_path) if other_path else rec
    keyed = other is not rec
    space = KeySpace(rec, other if keyed else None)
    pi, pj, d, turn = match_frames(rec, other, radius, angle)
    pairs = len(pi)
    print(f"previs frames {int((rec.frames['mode'] == 0).sum())}, CBRO frames {int(((other.frames['mode'] != 0) & (other.frames['cull'] != 0)).sum())} in {other.path.name}, matched pairs {pairs} (within {radius:.0f} units, {angle:.0f} deg)")
    if not pairs:
        print("no matched pairs: record a previs stretch and a CBRO stretch over the same ground (F8 in motion), or widen --radius/--angle")
        return

    def registered(r, frame_indices):
        # the distinct (pair, object key) of the main view's registrations over the matched frames, packed in one int64
        c, at = r.candidates_of(frame_indices)
        regs = c["origin"] == 5
        return np.unique((at[regs] << 32) | space.of(r, c["object"][regs]))

    a = registered(rec, pi)
    b = registered(other, pj)
    previs_only = a[~np.isin(a, b, assume_unique=True)]
    cbro_only = b[~np.isin(b, a, assume_unique=True)]

    # CBRO's own decision on the previs-only objects in the matched frame: its candidates naming those objects, by
    # (object, decision), counted over the pairs where the object was previs-only
    decisions = {}
    wanted = np.unique(previs_only & 0xFFFFFFFF)
    if len(wanted):
        keys_all = space.of(other, other.candidates["object"])
        hit = np.nonzero(np.isin(keys_all, wanted) & (other.frame_of >= 0))[0]
        order = np.argsort(other.frame_of[hit], kind="stable")
        hit = hit[order]
        hit_frames = other.frame_of[hit]
        lo = np.searchsorted(hit_frames, pj, "left")
        hi = np.searchsorted(hit_frames, pj, "right")
        sel = ranges(lo, hi - lo)                        # the hit rows of each pair's CBRO frame
        at = np.repeat(np.arange(pairs, dtype=np.int64), hi - lo)
        key = keys_all[hit[sel]]
        keep = np.isin((at << 32) | key, previs_only)
        rows = other.candidates[hit[sel[keep]]]
        key = key[keep]
        codes, code_of = np.unique(packed(rows, ("origin", "outcome", "verdict", "sun", "streak", "source", "path", "flags")), return_inverse=True)
        combos, counts = np.unique((key << 32) | code_of.astype(np.int64), return_counts=True)
        for combo, n in zip(combos.tolist(), counts.tolist()):
            decisions.setdefault(combo >> 32, []).append((codes[combo & 0xFFFFFFFF], n))

    def why(code):
        flags = int(code["flags"])
        judged = f"{rec.name('outcome', int(code['outcome']))}/{rec.name('verdict', int(code['verdict']))} sun {rec.name('sun', int(code['sun']))} streak {int(code['streak'])} via {rec.name('source', int(code['source']))}" if flags & (1 << 8) else "no verdict"
        extra = ", ".join(n for n in rec.flag_names(flags) if n not in ("judged", "left out"))
        return f"{rec.name('origin', int(code['origin']))}: {judged}, path {rec.name('path', int(code['path']))}" + (f" [{extra}]" if extra else "")

    def show(title, only):
        keys, counts = np.unique(only & 0xFFFFFFFF, return_counts=True)
        print(f"\n{title}: {len(keys)} objects over {pairs} matched frames")
        for key, n in most_common(keys, counts, 40):
            print(f"  {n:>4} frames  {space.describe(key)}")
            for code, m in sorted(decisions.get(int(key), []), key=lambda kv: -kv[1])[:3]:
                print(f"             CBRO: {why(code)} ({m}x)")

    show("drawn by previs, never registered by CBRO in the matched frame", previs_only)
    show("registered by CBRO, never drawn by previs in the matched frame", cbro_only)


def cmd_frame(rec, args):
    i = int(args[0])
    f = rec.frames[i]
    print(f"frame {i}: clock {int(f['clock'])} render {int(f['renderFrame'])} mode {rec.name('mode', int(f['mode']))} settled {int(f['settled'])} cull {int(f['cull'])} async {int(f['asyncValid'])} location {int(f['location'])}")
    print(f"  eye ({f['eye'][0]:.1f}, {f['eye'][1]:.1f}, {f['eye'][2]:.1f}) yaw {rec.yaw(f):.1f} zoom {f['zoom'][0]:.3f}x{f['zoom'][1]:.3f} | view {np.array2string(f['view'], precision=3)} dilate {f['dilateMove']:.1f}")
    print(f"  sun {rec.name('sunState', int(f['sunState']))} dir {np.array2string(f['sunDir'], precision=3)} reach {f['sunReach']:.0f} | depth frame {int(f['depthFrame'])} readback {int(f['readback'])} merged {int(f['mergedFrames'])} viewSpace {int(f['viewSpace'])} hi-z {int(f['hiZWidth0'])}x{int(f['hiZHeight0'])} blob {int(f['hiZBlob'])} | lamps {int(f['lampCount'])}")
    c = rec.frame_candidates(i)
    print(f"  candidates {len(c)} (overflow {int(f['overflow'])}): " + ", ".join(f"{rec.name('origin', int(o))} {int(n)}" for o, n in zip(*np.unique(c["origin"], return_counts=True))))
    left = c[(c["flags"] & (1 << 9)) != 0]
    print("  left out by path: " + ", ".join(f"{rec.name('path', int(p))} {int(n)}" for p, n in zip(*np.unique(left["path"], return_counts=True))))
    if rec.hiz is not None and f["hiZBlob"] >= 0:
        blob = rec.hiz[int(f["hiZBlob"])]
        print(f"  hi-z level {rec.hiz_level}: farthest min {blob[0].min():.4f} max {blob[0].max():.4f}, nearest min {blob[1].min():.4f} max {blob[1].max():.4f}")


def cmd_objects(rec, args):
    pattern = re.compile(args[0], re.IGNORECASE)

    def matches(offsets):  # each distinct string searched once
        unique, inverse = np.unique(np.asarray(offsets, dtype=np.int64), return_inverse=True)
        return np.array([bool(pattern.search(rec.string(int(o)))) for o in unique] + [False], dtype=bool)[inverse]

    hits = np.nonzero(matches(rec.objects["rtti"]) | matches(rec.objects["name"]))[0]
    print(f"{len(hits)} objects match")
    named = np.isin(rec.candidates["object"], rec.objects["object"][hits]) & (rec.frame_of >= 0)
    c = rec.candidates[named]
    row = rec.object_rows(c["object"])
    seen = np.unique((row << 32) | rec.frame_of[named].astype(np.int64))  # distinct (object, frame)
    objects, in_frames = np.unique(seen >> 32, return_counts=True)
    frames_of = dict(zip(objects.tolist(), in_frames.tolist()))
    table = packed(c, ("origin", "outcome", "verdict", "path", "flags"))
    codes, code_of = np.unique(table, return_inverse=True)
    combos, counts = np.unique((row << 32) | code_of.astype(np.int64), return_counts=True)
    by_object = {}
    for combo, n in zip(combos.tolist(), counts.tolist()):
        by_object.setdefault(combo >> 32, []).append((codes[combo & 0xFFFFFFFF], n))
    for i in hits[:40].tolist():
        o = rec.objects[i]
        key = int(o["object"])
        print(f"{rec.describe(key)} at ({o['bound'][0]:.0f},{o['bound'][1]:.0f},{o['bound'][2]:.0f}) r {o['bound'][3]:.0f}, flags {int(o['flags']):#x}, in {int(frames_of.get(i, 0))} frames")
        for code, n in sorted(by_object.get(i, []), key=lambda kv: -kv[1])[:5]:
            print(f"    {n:>5}x {rec.name('origin', int(code['origin']))} {rec.name('outcome', int(code['outcome']))}/{rec.name('verdict', int(code['verdict']))} path {rec.name('path', int(code['path']))} [{', '.join(rec.flag_names(int(code['flags'])))}]")


def cmd_check(rec, args):
    problems = []
    starts = rec.frames["candidateOffset"].astype(np.int64)
    counts = rec.frames["candidateCount"].astype(np.int64)
    expected = np.cumsum(counts) - counts
    for i in np.nonzero(starts != expected)[0].tolist():
        problems.append(f"frame {i}: candidates start at {starts[i]}, expected {expected[i]}")
    for i in np.nonzero(starts + counts > len(rec.candidates))[0].tolist():
        problems.append(f"frame {i}: candidates run past the section")
    for i in np.nonzero(rec.frames["lampOffset"].astype(np.int64) + rec.frames["lampCount"].astype(np.int64) > len(rec.lamps))[0].tolist():
        problems.append(f"frame {i}: lamps run past the section")
    blobs = rec.frames["hiZBlob"].astype(np.int64)
    for i in np.nonzero((blobs >= 0) & (blobs >= (0 if rec.hiz is None else len(rec.hiz))))[0].tolist():
        problems.append(f"frame {i}: hi-z blob {blobs[i]} missing")
    end = int(starts[-1] + counts[-1]) if len(rec.frames) else 0
    if end != len(rec.candidates):
        problems.append(f"candidates: {len(rec.candidates)} in the section, {end} owned by frames")
    for field in ("rtti", "name"):
        offsets = rec.objects[field].astype(np.int64)
        for i in np.nonzero(offsets >= len(rec.strings))[0].tolist():
            problems.append(f"object {i}: {field} offset {offsets[i]} past the string table")
    named = rec.candidates["object"][rec.candidates["object"] != 0]
    unknown = int((rec.object_rows(named) < 0).sum())
    if unknown:
        problems.append(f"{unknown} candidates name objects the objects section lacks")
    for p in problems[:20]:
        print("problem: " + p)
    print(f"check: {len(rec.frames)} frames, {len(rec.candidates)} candidates, {len(rec.objects)} objects, {len(rec.lamps)} lamps, {0 if rec.hiz is None else len(rec.hiz)} hi-z blobs: {'OK' if not problems else str(len(problems)) + ' problems'}")
    return 0 if not problems else 1


COMMANDS = {"summary": cmd_summary, "verdicts": cmd_verdicts, "origins": cmd_origins, "noverdict": cmd_noverdict, "diff": cmd_diff, "frame": cmd_frame, "objects": cmd_objects, "check": cmd_check}


def main(argv):
    if len(argv) < 1 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    rec = Recording(argv[0])
    command = argv[1] if len(argv) > 1 else "summary"
    if command not in COMMANDS:
        print(f"unknown command {command}\n{__doc__}")
        return 2
    result = COMMANDS[command](rec, argv[2:])
    return result or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
