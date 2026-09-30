param(
    [string]$Exe = "D:\Games\Fallout 4\Fallout4.exe",
    [string]$Db  = "D:\MO2\Fallout 4\mods\Address Library for F4SE Plugins\F4SE\Plugins\version-1-10-163-0.bin"
)

Add-Type -TypeDefinition @"
using System;
using System.IO;
using System.Collections.Generic;
using System.Text;

public class Fo4Image {
    public byte[] Data;
    public ulong ImageBase;
    struct Sec { public uint Va, VSize, Raw, RawSize; public string Name; }
    List<Sec> secs = new List<Sec>();
    public Dictionary<ulong, ulong> Ids = new Dictionary<ulong, ulong>();

    public Fo4Image(string exe, string db, ulong[] wanted) {
        Data = File.ReadAllBytes(exe);
        int pe = BitConverter.ToInt32(Data, 0x3C);
        int nsec = BitConverter.ToUInt16(Data, pe + 6);
        int optSize = BitConverter.ToUInt16(Data, pe + 20);
        int opt = pe + 24;
        ImageBase = BitConverter.ToUInt64(Data, opt + 24);
        int sh = opt + optSize;
        for (int i = 0; i < nsec; i++) {
            int o = sh + i * 40;
            secs.Add(new Sec {
                Name = Encoding.ASCII.GetString(Data, o, 8).TrimEnd('\0'),
                VSize = BitConverter.ToUInt32(Data, o + 8), Va = BitConverter.ToUInt32(Data, o + 12),
                RawSize = BitConverter.ToUInt32(Data, o + 16), Raw = BitConverter.ToUInt32(Data, o + 20) });
        }
        var want = new HashSet<ulong>(wanted);
        using (var br = new BinaryReader(File.OpenRead(db))) {
            ulong count = br.ReadUInt64();
            for (ulong i = 0; i < count; i++) {
                ulong id = br.ReadUInt64(), off = br.ReadUInt64();
                if (want.Contains(id)) Ids[id] = off;
            }
        }
    }

    public string Sections() {
        var sb = new StringBuilder();
        foreach (var s in secs) sb.AppendFormat("{0} va=0x{1:X} vsize=0x{2:X} raw=0x{3:X} rawsize=0x{4:X}\n", s.Name, s.Va, s.VSize, s.Raw, s.RawSize);
        return sb.ToString();
    }

    public long FileOff(ulong rva) {
        foreach (var s in secs) if (rva >= s.Va && rva < s.Va + Math.Max(s.VSize, s.RawSize)) return (long)(rva - s.Va + s.Raw);
        return -1;
    }
    public string SectionOf(ulong rva) {
        foreach (var s in secs) if (rva >= s.Va && rva < s.Va + Math.Max(s.VSize, s.RawSize)) return s.Name;
        return "?";
    }
    public byte U8(ulong rva) { return Data[FileOff(rva)]; }
    public int I32(ulong rva) { return BitConverter.ToInt32(Data, (int)FileOff(rva)); }
    public ulong U64(ulong rva) { return BitConverter.ToUInt64(Data, (int)FileOff(rva)); }
    public string Hex(ulong rva, int n) {
        long f = FileOff(rva); if (f < 0) return "(unmapped)";
        var sb = new StringBuilder();
        for (int i = 0; i < n; i++) sb.AppendFormat("{0:X2} ", Data[f + i]);
        return sb.ToString().Trim();
    }
    public string CStr(ulong rva) {
        long f = FileOff(rva); if (f < 0) return "(unmapped)";
        int e = (int)f; while (e < Data.Length && Data[e] != 0 && e - f < 128) e++;
        return Encoding.ASCII.GetString(Data, (int)f, e - (int)f);
    }
    // MSVC RTTI: vtable[-1] -> CompleteObjectLocator{sig, off, cdOff, typeDescRva, classDescRva, selfRva}
    public string MsvcName(ulong vtableRva) {
        try {
            ulong col = U64(vtableRva - 8) - ImageBase;
            uint td = (uint)I32(col + 12);
            return CStr(td + 16);
        } catch { return "(no COL)"; }
    }
    // Slot 2 of an NiObject vtable is GetRTTI: lea rax,[rip+disp]; ret
    public string RttiName(ulong vtableRva) {
        ulong fn = U64(vtableRva + 16) - ImageBase;
        if (U8(fn) != 0x48 || U8(fn + 1) != 0x8D || U8(fn + 2) != 0x05 || U8(fn + 7) != 0xC3) return "(GetRTTI pattern mismatch: " + Hex(fn, 8) + ")";
        ulong rtti = (ulong)((long)fn + 7 + I32(fn + 3));
        ulong namePtr = U64(rtti) - ImageBase;
        return CStr(namePtr);
    }
}
"@

$ids = @{
    'DrawWorld::Render_PreUI' = 984743; 'ForwardAlphaImpl' = 338205;
    'VT NiCamera' = 1305073; 'VT NiCullingProcess' = 547317; 'VT BSCullingProcess' = 556243;
    'VT BSGeometryListCullingProcess' = 1398386; 'VT BSParabolicCullingProcess' = 845854;
    'VT __MainCullingCamera' = 519680; 'VT BSCombinedTriShape' = 358895; 'VT BSMultiBoundNode' = 218331; 'VT BSFadeNodeCuller' = 1334420;
    'Main::Singleton' = 756304; 'Main::WorldRootCamera' = 384264; 'GridCellArray::Get' = 1330136;
    'BSGraphics::State' = 600795; 'RendererData ptr' = 1235449
}
$img = New-Object Fo4Image($Exe, $Db, [uint64[]]@($ids.Values))
"ImageBase=0x{0:X}" -f $img.ImageBase
$img.Sections()
foreach ($k in $ids.Keys | Sort-Object) {
    $id = [uint64]$ids[$k]
    if ($img.Ids.ContainsKey($id)) { "{0,-34} id={1,-8} rva=0x{2:X} ({3})" -f $k, $id, $img.Ids[$id], $img.SectionOf($img.Ids[$id]) } else { "{0,-34} id={1} MISSING" -f $k, $id }
}

"`n== call sites =="
$sites = @(
    @('prepass', 984743, 0x17F), @('hbao', 984743, 0x1BA), @('forward', 984743, 0x1C9),
    @('postResolveDepth', 338205, 0x1DC), @('firstPersonAlpha', 338205, 0x253), @('updateDynRes', 984743, 0x14B)
)
foreach ($s in $sites) {
    $rva = $img.Ids[[uint64]$s[1]] + [uint64]$s[2]
    $b = $img.U8($rva)
    $tgt = if ($b -eq 0xE8) { "0x{0:X}" -f ([int64]$rva + 5 + $img.I32($rva + 1)) } else { "-" }
    "{0,-18} rva=0x{1:X} bytes={2}  call->{3}" -f $s[0], $rva, $img.Hex($rva - 4, 12), $tgt
}

"`n== vtables =="
foreach ($k in 'VT NiCamera','VT NiCullingProcess','VT BSCullingProcess','VT BSGeometryListCullingProcess','VT BSParabolicCullingProcess','VT __MainCullingCamera','VT BSCombinedTriShape','VT BSMultiBoundNode') {
    $vt = $img.Ids[[uint64]$ids[$k]]
    $ni = try { $img.RttiName($vt) } catch { "(err)" }
    "{0,-32} rva=0x{1:X} msvc='{2}' niRTTI(slot2)='{3}'" -f $k, $vt, $img.MsvcName($vt), $ni
}
"NiCamera vtable first slots:"
$vt = $img.Ids[[uint64]1305073]
foreach ($slot in 0..4) { $v = $img.U64($vt + 8*$slot); "  [{0}] 0x{1:X}" -f $slot, $v }
"GetRTTI (slot 2) bytes:"
foreach ($k in 'VT NiCamera','VT BSCombinedTriShape','VT BSMultiBoundNode') {
    $vt = $img.Ids[[uint64]$ids[$k]]
    $fn = $img.U64($vt + 16) - $img.ImageBase
    $rtti = [int64]$fn + 7 + $img.I32($fn + 3)
    "  {0,-22} fn=0x{1:X} bytes={2} -> NiRTTI at rva 0x{3:X} ({4})" -f $k, $fn, $img.Hex($fn, 8), $rtti, $img.SectionOf([uint64]$rtti)
}

"IsNode slots 3/4 bytes:"
foreach ($k in 'VT BSMultiBoundNode','VT BSCombinedTriShape','VT NiCamera') {
    $vt = $img.Ids[[uint64]$ids[$k]]
    foreach ($slot in 3,4) { $fn = $img.U64($vt + 8*$slot) - $img.ImageBase; "  {0,-22} [{1}] {2}" -f $k, $slot, $img.Hex($fn, 6) }
}

"BSFadeNodeCuller slots:"
$vt = $img.Ids[[uint64]1334420]
"  msvc='{0}'" -f $img.MsvcName($vt)
foreach ($slot in 0x18..0x1D) { $v = $img.U64($vt + 8*$slot); $fn = $v - $img.ImageBase; $hex = if ($v -ge $img.ImageBase -and $v -lt $img.ImageBase + 0x7000000) { $img.Hex($fn, 20) } else { "(not code: 0x{0:X})" -f $v }; "  [0x{0:X2}] 0x{1:X} {2}" -f $slot, $v, $hex }
