"""Names the module+offset tokens in CBRO's sampler lines (bSampler, Core/Sampler): CBRO.dll from the linker map
(build/release/CBRO.map, /MAP), Fallout4.exe from the address database (fo4_database.csv) and ShaderReplacer.dll from
its PDB next to the installed DLL (dbghelp; nothing is disassembled). Prints every "sampler:" line of the log with each
token annotated as token[name+offset].

Given the sampler's histogram CSV instead (Data/CBRO_Recordings/CBRO-samples-<date>.csv, v1.96: every charged address
of every summary), rolls the samples up by function per summary, mode and stage: samples a frame, the share of the
stage, and milliseconds a frame at the sampling period (an estimate: the log's CPU-per-frame line is the measure).

With --lines (a CSV only), CBRO.dll's samples are rolled up by source line instead, from the build's PDB
(build/release/CBRO.pdb through dbghelp's line records: the profile's own symbols, nothing disassembled); the PDB
must be the one of the DLL that ran.

usage: python -I tools/sampler_names.py [path to CBRO.log | path to CBRO-samples-<date>.csv] [--top N] [--stage NAME] [--lines]
"""
import bisect
import csv
import ctypes
import os
import re
import sys
from ctypes import wintypes as w
from pathlib import Path

OPTS = sys.argv[1:]
TOP = int(OPTS[OPTS.index("--top") + 1]) if "--top" in OPTS else 40
STAGE = OPTS[OPTS.index("--stage") + 1] if "--stage" in OPTS else None
ARGS = [a for i, a in enumerate(OPTS) if not a.startswith("--") and (i == 0 or OPTS[i - 1] not in ("--top", "--stage"))]
LOG = Path(ARGS[0]) if ARGS else Path(os.path.expandvars(r"%USERPROFILE%\Documents\My Games\Fallout4\F4SE\CBRO.log"))
LINES = "--lines" in OPTS
PERIOD_MS = 0.4  # Core/Sampler kPeriod100ns
MAP = Path(__file__).resolve().parents[1] / "build" / "release" / "CBRO.map"
CSV = Path(r"D:\Projects\FO4OG\re\fo4_database.csv")
RR_DLL = Path(r"D:\MO2\Fallout 4\mods\Shader Replacer Framework\F4SE\Plugins\ShaderReplacer.dll")
EXE_BASE = 0x140000000
RR_BASE = 0x180000000


# ---- CBRO.dll: the linker map's "Publics by Value" -------------------------------------------------------------
def load_map(path):
    base = 0x180000000
    rows = []
    in_publics = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Preferred load address is" in line:
            base = int(line.split()[-1], 16)
        if "Publics by Value" in line:
            in_publics = True
            continue
        if not in_publics:
            continue
        parts = line.split()
        if len(parts) < 4 or ":" not in parts[0]:
            continue
        try:
            address = int(parts[2], 16)
        except ValueError:
            continue
        if address >= base:
            rows.append((address - base, parts[1]))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def undecorate(name):
    """The readable core of an MSVC decorated name: Scope::Func, Func::<lambda_1> for a lambda's operator()."""
    def scopes(text):
        return [s for s in text.split("@") if s and not s.startswith("?A0x")]  # ?A0x...: an anonymous namespace

    m = re.match(r"\?\?R(<lambda_\d+>)@\?\d\?\?(\w+)@((?:[\w?]+@)*)@", name)
    if m:
        return "::".join(list(reversed(scopes(m.group(3)))) + [m.group(2), m.group(1)])
    m = re.match(r"\?(\w+)@((?:[\w?]+@)*)@", name)
    if m:
        return "::".join(list(reversed(scopes(m.group(2)))) + [m.group(1)])
    return name


# ---- Fallout4.exe: the address database -----------------------------------------------------------------------
def load_csv(path):
    rows = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            try:
                address = int(row["fo4"], 16)
            except (ValueError, KeyError):
                continue
            if address and row.get("name"):
                rows.append((address - EXE_BASE, row["name"]))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def nearest(keys, names, offset):
    i = bisect.bisect_right(keys, offset) - 1
    return None if i < 0 else (names[i], offset - keys[i])


# ---- ShaderReplacer.dll: its PDB through dbghelp ----------------------------------------------------------------
class SYMBOL_INFOW(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", w.ULONG), ("TypeIndex", w.ULONG), ("Reserved", ctypes.c_ulonglong * 2), ("Index", w.ULONG),
        ("Size", w.ULONG), ("ModBase", ctypes.c_ulonglong), ("Flags", w.ULONG), ("Value", ctypes.c_ulonglong),
        ("Address", ctypes.c_ulonglong), ("Register", w.ULONG), ("Scope", w.ULONG), ("Tag", w.ULONG),
        ("NameLen", w.ULONG), ("MaxNameLen", w.ULONG), ("Name", ctypes.c_wchar * 2048),
    ]


class IMAGEHLP_LINEW64(ctypes.Structure):
    _fields_ = [("SizeOfStruct", w.DWORD), ("Key", ctypes.c_void_p), ("LineNumber", w.DWORD), ("FileName", w.LPWSTR), ("Address", ctypes.c_ulonglong)]


class PdbNames:
    def __init__(self, dll, base=RR_BASE, handle=0x4242):
        self.base = base
        self.ok = False
        if not dll.exists():
            return
        self.dbghelp = ctypes.WinDLL("dbghelp", use_last_error=True)
        d = self.dbghelp
        d.SymInitializeW.argtypes = [w.HANDLE, w.LPCWSTR, w.BOOL]
        d.SymInitializeW.restype = w.BOOL
        d.SymLoadModuleExW.argtypes = [w.HANDLE, w.HANDLE, w.LPCWSTR, w.LPCWSTR, ctypes.c_ulonglong, w.DWORD, ctypes.c_void_p, w.DWORD]
        d.SymLoadModuleExW.restype = ctypes.c_ulonglong
        d.SymFromAddrW.argtypes = [w.HANDLE, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(SYMBOL_INFOW)]
        d.SymFromAddrW.restype = w.BOOL
        d.SymGetLineFromAddrW64.argtypes = [w.HANDLE, ctypes.c_ulonglong, ctypes.POINTER(w.DWORD), ctypes.POINTER(IMAGEHLP_LINEW64)]
        d.SymGetLineFromAddrW64.restype = w.BOOL
        d.SymSetOptions(0x2 | 0x10 | 0x40 | 0x2000)  # UNDNAME, LOAD_LINES, LOAD_ANYTHING, IGNORE_NT_SYMPATH
        self.proc = w.HANDLE(handle)
        if not d.SymInitializeW(self.proc, str(dll.parent), False):
            print(f"warning: SymInitialize failed ({ctypes.get_last_error()}): ShaderReplacer offsets left unnamed", file=sys.stderr)
            return
        if not d.SymLoadModuleExW(self.proc, None, str(dll), None, base, dll.stat().st_size, None, 0):
            print(f"warning: SymLoadModuleEx failed ({ctypes.get_last_error()}): ShaderReplacer offsets left unnamed", file=sys.stderr)
            return
        self.ok = True

    def name(self, offset):
        if not self.ok:
            return None
        sym = SYMBOL_INFOW()
        sym.SizeOfStruct = 88
        sym.MaxNameLen = 2047
        disp = ctypes.c_ulonglong()
        if self.dbghelp.SymFromAddrW(self.proc, self.base + offset, ctypes.byref(disp), ctypes.byref(sym)):
            return (sym.Name, disp.value)
        return None

    def line(self, offset):
        """(file name, line number) of the source line an offset's instruction came from, or None."""
        if not self.ok:
            return None
        rec = IMAGEHLP_LINEW64()
        rec.SizeOfStruct = ctypes.sizeof(IMAGEHLP_LINEW64)
        disp = w.DWORD()
        if self.dbghelp.SymGetLineFromAddrW64(self.proc, self.base + offset, ctypes.byref(disp), ctypes.byref(rec)):
            return (Path(rec.FileName).name, rec.LineNumber)
        return None


map_keys, map_names = load_map(MAP) if MAP.exists() else ([], [])
csv_keys, csv_names = load_csv(CSV) if CSV.exists() else ([], [])
rr = PdbNames(RR_DLL)
cbro = PdbNames(MAP.with_suffix(".dll"), 0x180000000, 0x4343) if LINES else None
token = re.compile(r"(CBRO\.dll|Fallout4\.exe|ShaderReplacer\.dll)\+([0-9A-Fa-f]+)")


def annotate(m):
    module, offset = m.group(1), int(m.group(2), 16)
    if module == "CBRO.dll":
        hit = nearest(map_keys, map_names, offset)
        hit = (undecorate(hit[0]), hit[1]) if hit else None
    elif module == "Fallout4.exe":
        hit = nearest(csv_keys, csv_names, offset)
    else:
        hit = rr.name(offset)
    return f"{m.group(0)}[{hit[0]}+{hit[1]:X}]" if hit else m.group(0)


def function_of(where):
    """The function (or, without symbols, the module) a module+offset token belongs to."""
    m = token.fullmatch(where)
    if not m:
        return where.split("+")[0] if "+" in where else where
    module, offset = m.group(1), int(m.group(2), 16)
    if module == "CBRO.dll":
        hit = nearest(map_keys, map_names, offset)
        return f"CBRO.dll {undecorate(hit[0])}" if hit else where
    if module == "Fallout4.exe":
        hit = nearest(csv_keys, csv_names, offset)
        return f"Fallout4.exe {hit[0]}" if hit else where
    hit = rr.name(offset)
    return f"ShaderReplacer.dll {hit[0]}" if hit else where


def rollup_csv(path):
    from collections import defaultdict
    stages = {}  # (summary, mode, stage) -> {function: samples}
    frames = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["summary"]), row["mode"], row["stage"])
            if STAGE and row["stage"] != STAGE:
                continue
            frames[key] = max(1, int(row["frames"]))
            name = function_of(row["where"])
            if LINES:
                m = token.fullmatch(row["where"])
                if not m or m.group(1) != "CBRO.dll":
                    continue
                hit = cbro.line(int(m.group(2), 16))
                name = f"{hit[0]}:{hit[1]}  ({name.removeprefix('CBRO.dll ')})" if hit else name
            stages.setdefault(key, defaultdict(int))[name] += int(row["samples"])
    for key in sorted(stages):
        summary, mode, stage = key
        funcs = stages[key]
        total = sum(funcs.values())
        n = frames[key]
        print(f"\n==== summary {summary} ({n} frames) | {mode} / {stage}: {total} samples listed, {total / n:.1f} a frame, ~{total * PERIOD_MS / n:.2f} ms a frame ====")
        print(f"{'samples/frame':>13} {'share':>6} {'~ms/frame':>9}  function")
        for name, count in sorted(funcs.items(), key=lambda kv: -kv[1])[:TOP]:
            print(f"{count / n:>13.1f} {100.0 * count / total:>5.1f}% {count * PERIOD_MS / n:>9.3f}  {name}")


def annotate_log(path):
    rollup = re.compile(r"CBRO\.dll\+[0-9A-Fa-f]+\[(.+?)\+[0-9A-Fa-f]+\] ([0-9.]+)%")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "sampler:" not in line:
            continue
        named = token.sub(annotate, line)
        print(named)
        if "CBRO.dll by address" in named:
            # the stage's CBRO.dll samples summed per function, largest first
            shares = {}
            for name, share in rollup.findall(named):
                shares[name] = shares.get(name, 0.0) + float(share)
            print("    by function: " + ", ".join(f"{name} {share:.1f}%" for name, share in sorted(shares.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":  # importable for its symbol resolution (function_of) by other offline scripts
    if LOG.suffix.lower() == ".csv":
        rollup_csv(LOG)
    else:
        annotate_log(LOG)
