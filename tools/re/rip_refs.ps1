param([string[]]$Targets)
$scratch = Split-Path $MyInvocation.MyCommand.Path
$null = & "$scratch\xref_scan.ps1" -Targets @('dummy=1')
$x = New-Object Xref("D:\Games\Fallout 4\Fallout4.exe", "D:\MO2\Fallout 4\mods\Address Library for F4SE Plugins\F4SE\Plugins\version-1-10-163-0.bin")
$t = @(); $l = @()
foreach ($pair in $Targets) { $parts = $pair.Split('='); $l += $parts[0]; $t += [uint32]([Convert]::ToUInt32($parts[1], 16)) }
$x.RipRefs([uint32[]]$t, [string[]]$l)
