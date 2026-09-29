param([string[]]$Sites, [int]$Span = 0x800)

$scratch = Split-Path $MyInvocation.MyCommand.Path
. {
    # Reuse the Xref type (compiles once per session)
    $null = & "$scratch\xref_scan.ps1" -Targets @('dummy=1')
}

# OG id -> name from CommonLibF4RD's id headers (vtables, MSVC RTTI, NiRTTI; entries are REL::ID(og, ae)).
# Function ids have no name table any more (CommonLibF4-DM's IDs.h was removed with that library); the $extra map below covers the ones CBRO uses.
$rd = "D:\Projects\Confidence-Based Runtime Occlusion System\external\CommonLibF4RD\CommonLibF4\include\RE"
$names = @{}
foreach ($table in @(@{ file = "$rd\VTABLE_IDs.h"; ns = 'VTABLE' }, @{ file = "$rd\RTTI_IDs.h"; ns = 'RTTI' }, @{ file = "$rd\NiRTTI_IDs.h"; ns = 'NiRTTI' })) {
    if (-not (Test-Path $table.file)) { continue }
    foreach ($line in Get-Content $table.file) {
        if ($line -match '(\w+)\s*\{\s*REL::ID\(\s*(\d+)') {
            $id = [uint64]$Matches[2]
            if (-not $names.ContainsKey($id)) { $names[$id] = "$($table.ns)::$($Matches[1])" }
        }
    }
}
# fo4test / plan names
$extra = @{ 984743='DrawWorld::Render_PreUI'; 587723='DrawWorld::Imagespace'; 338205='ForwardAlphaImpl'; 656535='DrawWorld::FrameGenerationForward(hooked)';
            728427='DrawWorld::DeferredComposite'; 423200='BSCullingProcess::ctor'; 125924='BSGeometryListCullingProcess::ctor'; 1288195='BSParabolicCullingProcess::ctor';
            708657='MainCullingCamera::?'; 1281872='TES::UpdateMultiBoundVisibility'; 88488='Interface3D::Renderer::Create' }
foreach ($k in $extra.Keys) { $names[[uint64]$k] = $extra[$k] }

$x = New-Object Xref("D:\Games\Fallout 4\Fallout4.exe", "D:\MO2\Fallout 4\mods\Address Library for F4SE Plugins\F4SE\Plugins\version-1-10-163-0.bin")
foreach ($k in $names.Keys) { $x.IdNames[$k] = $names[$k] }

foreach ($s in $Sites) {
    $rva = [Convert]::ToUInt32($s, 16)
    $fn = $x.Primary($rva)
    "== site 0x{0:X} -> primary fn {1}" -f $rva, $x.Label($fn)
    $x.CallsInAllChunks($fn)
}
