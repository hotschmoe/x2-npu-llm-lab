# Build a DSP-only diag variant and package it: d2-variant.ps1 -Name x -Defines "A=1;B=2"  (host dll = current build-decode2 dll)
param([Parameter(Mandatory=$true)][string]$Name, [string]$Defines = '')
$ErrorActionPreference = 'Continue'
$out = & cmd /c "`"$PSScriptRoot\d2-skel.cmd`" `"$Defines`"" 2>&1
$last = $out | Select-Object -Last 2
if (($out -join "`n") -notmatch 'BUILD_EXIT=0') { $out | Select-Object -Last 30; throw 'skel build failed' }
& "$PSScriptRoot\pkg-decode2.ps1" -Name $Name | Select-Object -Last 1
