param(
    [string]$Exe = "D:\Games\Fallout 4\Fallout4.exe",
    [string]$Db  = "D:\MO2\Fallout 4\mods\Address Library for F4SE Plugins\F4SE\Plugins\version-1-10-163-0.bin",
    [string[]]$Targets
)

Add-Type -TypeDefinition @"
using System;
using System.IO;
using System.Collections.Generic;
using System.Text;

public class Xref {
    public byte[] D; public ulong Base;
    struct Sec { public uint Va, VSize, Raw, RawSize; public string Name; }
    List<Sec> secs = new List<Sec>();
    public Dictionary<ulong, ulong> RvaToId = new Dictionary<ulong, ulong>();
    uint[] fnBegin; uint[] fnEnd;

    public Xref(string exe, string db) {
        D = File.ReadAllBytes(exe);
        int pe = BitConverter.ToInt32(D, 0x3C);
        int nsec = BitConverter.ToUInt16(D, pe + 6);
        int optSize = BitConverter.ToUInt16(D, pe + 20);
        int opt = pe + 24;
        Base = BitConverter.ToUInt64(D, opt + 24);
        int sh = opt + optSize;
        for (int i = 0; i < nsec; i++) {
            int o = sh + i * 40;
            secs.Add(new Sec { Name = Encoding.ASCII.GetString(D, o, 8).TrimEnd('\0'),
                VSize = BitConverter.ToUInt32(D, o + 8), Va = BitConverter.ToUInt32(D, o + 12),
                RawSize = BitConverter.ToUInt32(D, o + 16), Raw = BitConverter.ToUInt32(D, o + 20) });
        }
        // exception directory (index 3) -> RUNTIME_FUNCTION table
        uint excRva = ExcRva = BitConverter.ToUInt32(D, opt + 112 + 3 * 8), excSize = BitConverter.ToUInt32(D, opt + 112 + 3 * 8 + 4);
        int n = (int)(excSize / 12); fnBegin = new uint[n]; fnEnd = new uint[n];
        long eo = Off(excRva);
        for (int i = 0; i < n; i++) { fnBegin[i] = BitConverter.ToUInt32(D, (int)(eo + i * 12)); fnEnd[i] = BitConverter.ToUInt32(D, (int)(eo + i * 12 + 4)); }
        using (var br = new BinaryReader(File.OpenRead(db))) {
            ulong cnt = br.ReadUInt64();
            for (ulong i = 0; i < cnt; i++) { ulong id = br.ReadUInt64(), off = br.ReadUInt64(); if (!RvaToId.ContainsKey(off)) RvaToId[off] = id; }
        }
    }
    public long Off(ulong rva) { foreach (var s in secs) if (rva >= s.Va && rva < s.Va + s.RawSize) return (long)(rva - s.Va + s.Raw); return -1; }

    // Innermost RUNTIME_FUNCTION containing rva; chained unwind entries are resolved to their primary function.
    public uint FuncOf(uint rva) {
        int lo = 0, hi = fnBegin.Length - 1, best = -1;
        while (lo <= hi) { int mid = (lo + hi) / 2; if (fnBegin[mid] <= rva) { best = mid; lo = mid + 1; } else hi = mid - 1; }
        if (best < 0 || rva >= fnEnd[best]) return 0;
        return fnBegin[best];
    }
    public string Name(uint rva) {
        if (rva == 0) return "?";
        ulong id; return RvaToId.TryGetValue(rva, out id) ? String.Format("0x{0:X}(id {1})", rva, id) : String.Format("0x{0:X}(no id)", rva);
    }

    // Follows UNW_FLAG_CHAININFO to the primary function start.
    public uint Primary(uint rva) {
        uint fn = FuncOf(rva);
        for (int guard = 0; guard < 8 && fn != 0; guard++) {
            int idx = Array.BinarySearch(fnBegin, fn); if (idx < 0) break;
            long eo = Off(ExcRva + (uint)idx * 12 + 8); uint unwind = BitConverter.ToUInt32(D, (int)eo);
            long uo = Off(unwind); byte verFlags = D[uo]; byte codes = D[uo + 2];
            if (((verFlags >> 3) & 4) == 0) break;
            long chained = uo + 4 + ((codes + 1) & ~1) * 2;
            fn = BitConverter.ToUInt32(D, (int)chained);
        }
        return fn;
    }
    public uint ExcRva;
    public Dictionary<ulong, string> IdNames = new Dictionary<ulong, string>();
    public string Label(uint rva) {
        ulong id; if (!RvaToId.TryGetValue(rva, out id)) return String.Format("0x{0:X}", rva);
        string n; return IdNames.TryGetValue(id, out n) ? String.Format("0x{0:X} id {1} {2}", rva, id, n) : String.Format("0x{0:X} id {1}", rva, id);
    }
    public List<string> CallsIn(uint fnStart, uint fnLimit) {
        var r = new List<string>();
        int fi = Array.BinarySearch(fnBegin, fnStart); if (fi >= 0 && fnEnd[fi] < fnLimit) fnLimit = fnEnd[fi];
        r.Add(String.Format("  (scanning 0x{0:X}..0x{1:X})", fnStart, fnLimit));
        for (uint rva = fnStart; rva < fnLimit; rva++) {
            long o = Off(rva); if (D[o] != 0xE8) continue;
            uint tgt = (uint)(rva + 5 + BitConverter.ToInt32(D, (int)o + 1));
            if (Primary(tgt) != tgt) continue;   // only calls that land on a function start
            r.Add(String.Format("  +0x{0:X}: call {1}", rva - fnStart, Label(tgt)));
        }
        return r;
    }
    public List<string> CallsInAllChunks(uint fn) {
        var r = new List<string>();
        for (int i = 0; i < fnBegin.Length; i++) {
            if (fnBegin[i] != fn && (fnBegin[i] < fn || fnBegin[i] > fn + 0x10000)) continue;
            if (Primary(fnBegin[i]) != fn) continue;
            r.AddRange(CallsIn(fnBegin[i], fnEnd[i]));
        }
        return r;
    }
    // Any disp32 that resolves (as rip-relative, no trailing immediate) to one of the targets.
    public List<string> RipRefs(uint[] targets, string[] labels) {
        var set = new Dictionary<uint, string>();
        for (int i = 0; i < targets.Length; i++) set[targets[i]] = labels[i];
        var result = new List<string>();
        Sec text = secs.Find(s => s.Name == ".text");
        for (long o = text.Raw; o < text.Raw + text.RawSize - 4; o++) {
            uint dispAt = (uint)(o - text.Raw + text.Va);
            uint tgt = (uint)(dispAt + 4 + BitConverter.ToInt32(D, (int)o));
            string label; if (!set.TryGetValue(tgt, out label)) continue;
            uint fn = Primary(dispAt);
            result.Add(String.Format("{0,-24} ref at 0x{1:X} in fn {2}", label, dispAt, Label(fn)));
        }
        return result;
    }
    public List<string> Scan(uint[] targets, string[] labels) {
        var set = new Dictionary<uint, string>();
        for (int i = 0; i < targets.Length; i++) set[targets[i]] = labels[i];
        var result = new List<string>();
        Sec text = secs.Find(s => s.Name == ".text");
        long start = text.Raw, end = text.Raw + text.RawSize - 8;
        for (long o = start; o < end; o++) {
            byte b = D[o];
            uint here = (uint)(o - text.Raw + text.Va);
            string kind = null; uint tgt = 0;
            if (b == 0xE8 || b == 0xE9) {
                tgt = (uint)(here + 5 + BitConverter.ToInt32(D, (int)o + 1));
                kind = b == 0xE8 ? "call" : "jmp";
            } else if ((b == 0x48 || b == 0x4C) && D[o + 1] == 0x8D && (D[o + 2] & 0xC7) == 0x05) {
                tgt = (uint)(here + 7 + BitConverter.ToInt32(D, (int)o + 3));
                kind = "lea";
            } else continue;
            string label;
            if (!set.TryGetValue(tgt, out label)) continue;
            uint fn = FuncOf(here);
            result.Add(String.Format("{0,-28} {1,-4} at 0x{2:X} in fn {3} +0x{4:X}", label, kind, here, Name(fn), fn == 0 ? 0 : here - fn));
        }
        return result;
    }
}
"@

$x = New-Object Xref($Exe, $Db)
$t = @(); $l = @()
foreach ($pair in $Targets) { $parts = $pair.Split('='); $l += $parts[0]; $t += [uint32]([Convert]::ToUInt32($parts[1], 16)) }
$x.Scan([uint32[]]$t, [string[]]$l)
