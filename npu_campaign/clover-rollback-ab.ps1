# Interleaved A/B (ABAB...) of clover-run.ps1 over packages and prompts, one run per NPU-lock hold.
# clover-run.ps1 takes the NPU lock for every run (CPU runs too: unified LPDDR), so each run is one lock hold.
# Each run also dumps the committed token ids to clover-logs\<label>.tokens (LLAMA_SPEC_TOKENS_OUT).
# Usage: clover-rollback-ab.ps1 -Tag t1 -Pkgs @('A=npu_campaign\pkg-clover-rollback-prof0','B=npu_campaign\pkg-clover-rollback-rs2|LLAMA_SPEC_BONUS=1|DraftMax=7') -Prompts @('diffusion','code','math') -Rounds 2
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][string[]]$Pkgs,
    [string[]]$Prompts = @('diffusion'),
    [int]$Rounds = 2,
    [string]$Device = 'none',
    [int]$Threads = 12,
    [int]$DraftMax = 15,
    [int]$NPredict = 256,
    [string]$Accept = '',
    [string]$Extra = '-v'
)
$ErrorActionPreference = 'Stop'
if ($Tag -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid tag' }
$logDir = "$PSScriptRoot\clover-logs"
$rows = @()
$knobs = @('LLAMA_SPEC_BONUS', 'LLAMA_SPEC_VERIFY_MAX', 'LLAMA_DRAFT_SELF_CKPT', 'LLAMA_RS_ROLLBACK_NEGCTL', 'LLAMA_DRAFT_SELF_MARGIN')
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($p in $Prompts) {
        # alternate the order every round so neither side always runs first
        $order = if ($r % 2 -eq 1) { $Pkgs } else { $Pkgs[($Pkgs.Count - 1)..0] }
        foreach ($kv in $order) {
            # config = name=pkg[|ENV=value|...|DraftMax=n]; research env knobs are cleared before every run
            $parts = $kv.Split('|')
            $name, $pkg = $parts[0].Split('=', 2)
            foreach ($k in $knobs) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
            $dm = $DraftMax
            $exe = 'llama-speculative-simple.exe'   # Exe=llama-completion.exe: plain causal greedy decode reference
            foreach ($e in ($parts | Select-Object -Skip 1)) {
                $k, $v = $e.Split('=', 2)
                if ($k -eq 'DraftMax') { $dm = [int]$v } elseif ($k -eq 'Exe') { $exe = $v } else { Set-Item "env:$k" $v }
            }
            $label = "rb-$Tag-$name-$p-d$dm-r$r"
            $tok = "$logDir\$label.tokens"
            $env:LLAMA_SPEC_TOKENS_OUT = $tok
            $args_ = @{ Label = $label; Pkg = $pkg; Device = $Device; Threads = $Threads; Accept = $Accept; DraftMax = $dm;
                        NPredict = $NPredict; Prompt = "npu_campaign\clover-prompts\$p.txt"; Extra = $Extra; Exe = $exe }
            if ($exe -ne 'llama-speculative-simple.exe') { $args_.Extra = "$Extra -no-cnv" }
            # sample other model processes (and compilers: a build steals CPU / DRAM bandwidth) every 2 s and flag any that run
            # while ours does
            $ovf = "$logDir\$label.overlap"
            Remove-Item $ovf -ErrorAction SilentlyContinue
            $watch = Start-Job -ArgumentList $ovf -ScriptBlock {
                param($f)
                while ($true) {
                    $t = [int](Get-Date -UFormat %s)
                    Get-CimInstance Win32_Process -Filter "Name like 'llama-%' or Name like 'test-backend%' or Name like 'clang%' or Name = 'cl.exe' or Name like '%link.exe' or Name = 'ninja.exe'" |
                        ForEach-Object { "$t $($_.ProcessId) $([int]($_.CommandLine -like '*pkg-clover-rollback-*')) $($_.Name)" } | Add-Content $f
                    Start-Sleep -Seconds 2
                }
            }
            # npu-lock.ps1's stale-owner check can throw while another agent rewrites owner.txt (before any run starts,
            # nothing to clean up): retry those failures
            try {
                for ($try = 1; ; $try++) {
                    try { $out = & (Join-Path $PSScriptRoot 'clover-run.ps1') @args_; break }
                    catch {
                        if ($try -ge 50 -or $_.Exception.Message -notmatch 'Cannot convert|being used by another process') { throw }
                        Start-Sleep -Milliseconds 700
                    }
                }
            } finally { Stop-Job $watch; Remove-Job $watch -Force }
            $cpu = ($out | Where-Object { $_ -match 'HOST_CPU' }) -replace '.*= ([\d.]+).*', '$1'
            $err = Get-Content "$logDir\$label.err" -Raw
            $tps = if ($err -match 'decoded\s+(\d+) tokens in\s+([\d.]+) seconds, speed:\s+([\d.]+)') { $matches[3] }
                   elseif ($err -match 'eval time =\s+[\d.]+ ms /\s+\d+ runs.*?([\d.]+) tokens per second') { $matches[1] } else { 'NA' }
            # generation window ~ [end of .err - decode seconds - 3 s, end of .err]; foreign processes inside it make the row
            # OVERLAP, foreign processes only during model load / teardown make it 'loadovl'
            $dsec = if ($err -match 'decoded\s+\d+ tokens in\s+([\d.]+) seconds') { [double]$matches[1] }
                    elseif ($err -match 'eval time =\s+([\d.]+) ms /') { [double]$matches[1] / 1000 } else { 1e9 }
            $ge = [int](Get-Date (Get-Item "$logDir\$label.err").LastWriteTime -UFormat %s); $gs = $ge - [int]$dsec - 3
            $smp = @(Get-Content $ovf -ErrorAction SilentlyContinue | ForEach-Object { $a = $_.Split(' '); [pscustomobject]@{ t = [int]$a[0]; pid = $a[1]; mine = $a[2]; name = $a[3] } })
            $mineT = @($smp | Where-Object { $_.mine -eq '1' } | ForEach-Object { $_.t }) | Sort-Object -Unique
            $fsmp = @($smp | Where-Object { $_.mine -ne '1' -and $mineT -contains $_.t })
            $fgen = @($fsmp | Where-Object { $_.t -ge $gs -and $_.t -le $ge } | ForEach-Object { "$($_.name):$($_.pid)" }) | Sort-Object -Unique
            $fall = @($fsmp | ForEach-Object { "$($_.name):$($_.pid)" }) | Sort-Object -Unique
            $ovl = if ($fgen.Count) { "OVERLAP(" + ($fgen -join ',') + ")" } elseif ($fall.Count) { "loadovl(" + ($fall -join ',') + ")" } else { 'clean' }
            $fwd = "$((($err -split "`n") | Where-Object { $_ -match 'forwards/token' } | Select-Object -First 1) -replace '.*forwards: ', '')"
            $acc = if ($err -match '#mean acc len = ([\d.]+)') { $matches[1] } else { 'NA' }
            $row = "{0,-48} tok/s={1,8} meanacc={2,6} hostcpu={3,5} {4} {5}" -f $label, $tps, $acc, $cpu, $ovl, $fwd.Trim()
            $row
            $rows += $row
        }
    }
}
$rows | Set-Content "$logDir\rb-$Tag.summary.txt"
