param(
    [string]$Exe = "D:\Games\Fallout 4\Fallout4.exe",
    [string]$Db  = "D:\MO2\Fallout 4\mods\Address Library for F4SE Plugins\F4SE\Plugins\version-1-10-163-0.bin",
    [string]$Pattern = "Cull|Occlu|Visib|Frustum|Portal|MultiBound|PreCull|Combined"
)

Add-Type -TypeDefinition @"
using System;
using System.IO;
using System.Collections.Generic;
using System.Text;

public class RttiScan {
    public byte[] D; public ulong Base;
    struct Sec { public uint Va, VSize, Raw, RawSize; public string Name; }
    List<Sec> secs = new List<Sec>();
    public Dictionary<ulong, ulong> RvaToId = new Dictionary<ulong, ulong>();

    public RttiScan(string exe, string db) {
        D = File.ReadAllBytes(exe);
        int pe = BitConverter.ToInt32(D, 0x3C);
        int nsec = BitConverter.ToUInt16(D, pe + 6);
        int optSize = BitConverter.ToUInt16(D, pe + 20);
        Base = BitConverter.ToUInt64(D, pe + 24 + 24);
        int sh = pe + 24 + optSize;
        for (int i = 0; i < nsec; i++) {
            int o = sh + i * 40;
            secs.Add(new Sec { Name = Encoding.ASCII.GetString(D, o, 8).TrimEnd('\0'),
                VSize = BitConverter.ToUInt32(D, o + 8), Va = BitConverter.ToUInt32(D, o + 12),
                RawSize = BitConverter.ToUInt32(D, o + 16), Raw = BitConverter.ToUInt32(D, o + 20) });
        }
        using (var br = new BinaryReader(File.OpenRead(db))) {
            ulong n = br.ReadUInt64();
            for (ulong i = 0; i < n; i++) { ulong id = br.ReadUInt64(), off = br.ReadUInt64(); if (!RvaToId.ContainsKey(off)) RvaToId[off] = id; }
        }
    }
    long Off(ulong rva) { foreach (var s in secs) if (rva >= s.Va && rva < s.Va + s.RawSize) return (long)(rva - s.Va + s.Raw); return -1; }
    uint U32(ulong rva) { long o = Off(rva); return o < 0 ? 0 : BitConverter.ToUInt32(D, (int)o); }
    ulong U64(ulong rva) { long o = Off(rva); return o < 0 ? 0 : BitConverter.ToUInt64(D, (int)o); }
    string CStr(ulong rva) { long o = Off(rva); if (o < 0) return ""; int e = (int)o; while (e < D.Length && D[e] != 0 && e - o < 256) e++; return Encoding.ASCII.GetString(D, (int)o, e - (int)o); }

    public List<string> Run(string pattern, string mustDeriveFrom) {
        var re = new System.Text.RegularExpressions.Regex(pattern);
        var result = new List<string>();
        Sec rdata = secs.Find(s => s.Name == ".rdata");
        // COL -> list of vtable RVAs referencing it
        var colToVt = new Dictionary<ulong, List<ulong>>();
        for (ulong rva = rdata.Va; rva + 8 <= rdata.Va + rdata.RawSize; rva += 8) {
            ulong v = U64(rva);
            if (v < Base + rdata.Va || v >= Base + rdata.Va + rdata.RawSize) continue;
            ulong col = v - Base;
            if (U32(col) != 1 || U32(col + 20) != (uint)col) continue;   // signature 1, pSelf
            List<ulong> l; if (!colToVt.TryGetValue(col, out l)) { l = new List<ulong>(); colToVt[col] = l; }
            l.Add(rva + 8);
        }
        foreach (var kv in colToVt) {
            ulong col = kv.Key;
            uint offset = U32(col + 4);
            string name = CStr(U32(col + 12) + 16);
            ulong chd = U32(col + 16);
            uint nBases = U32(chd + 8); ulong arr = U32(chd + 12);
            var bases = new List<string>();
            for (uint i = 1; i < nBases && i < 64; i++) bases.Add(CStr(U32(U32(arr + 4 * i)) + 16));
            bool derives = mustDeriveFrom == null || bases.Exists(b => b.Contains(mustDeriveFrom)) || name.Contains(mustDeriveFrom);
            if (!derives && !re.IsMatch(name)) continue;
            foreach (var vt in kv.Value) {
                ulong id; string ids = RvaToId.TryGetValue(vt, out id) ? id.ToString() : "-";
                result.Add(String.Format("{0}{1} off=0x{2:X} vt=0x{3:X} id={4} bases=[{5}]", derives ? "* " : "  ", name, offset, vt, ids, String.Join(", ", bases)));
            }
        }
        result.Sort((a, b) => String.CompareOrdinal(a.Substring(2), b.Substring(2)));
        return result;
    }
}
"@

$scan = New-Object RttiScan($Exe, $Db)
$scan.Run($Pattern, ".?AVNiCullingProcess@@")
