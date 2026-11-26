param([switch]$PublicPolicyValueOnly)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$python=(Get-Command python -ErrorAction Stop).Source

function Assert-PublicPolicyValueRejectsOverflow($engine,$rules,$model,$log,$budget,$scratch) {
    $extreme=$model+'.overflow'
    $opening=@(& $engine replay-public-policy-value $rules $extreme $log 1)
    if($LASTEXITCODE -ne 0 -or $opening.Count -ne 1 -or (ConvertFrom-Json $opening[0]).ply -ne 0){
        throw 'Delayed public model overflow fixture failed at the opening'
    }
    $stdout=Join-Path $scratch 'overflow-stdout.txt'
    $stderr=Join-Path $scratch 'overflow-stderr.txt'
    $arguments=@('replay-public-policy-value',('"{0}"' -f $rules),('"{0}"' -f $extreme),
                 ('"{0}"' -f $log),[string]$budget)
    $child=Start-Process -FilePath $engine -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if($child.ExitCode -ne 1 -or (Get-Item -LiteralPath $stdout).Length -ne 0 -or
       (Get-Content -Raw -LiteralPath $stderr) -notmatch 'non-finite public policy projection'){
        throw 'Public policy/value overflow must fail without emitting partial JSON'
    }
    Write-Output 'PASS late public model overflow rejected with zero stdout bytes'
}

function Invoke-PublicPolicyValueCheck {
    $scratch=Join-Path ([IO.Path]::GetTempPath()) ('junqi-pv-'+[guid]::NewGuid().ToString())
    [IO.Directory]::CreateDirectory($scratch) | Out-Null
    Write-Output ('Public policy/value artifacts: '+$scratch)
    $fixture=Join-Path $root 'build\core_smoke.exe'
    $engine=Join-Path $root 'build\junqi.exe'
    $rules=Join-Path $root 'config\ruleset-selfplay.json'
    $observation=Join-Path $root 'config\observation.json'
    $model=Join-Path $scratch 'model.bin'
    $training=& $fixture --public-policy-value-fixtures $rules $model
    $training | Write-Output
    if($LASTEXITCODE -ne 0){throw 'Public policy/value fixtures failed'}
    $seeds=@(85927,85931)
    $budget=128
    $count=0
    $clock=[Diagnostics.Stopwatch]::StartNew()
    foreach($seed in $seeds) {
        $log=Join-Path $scratch ('dev-'+$seed+'.jsonl')
        & $engine simulate $seed $log $rules $observation
        if($LASTEXITCODE -ne 0){throw 'Public policy/value development replay generation failed'}
        $scores=@(& $engine replay-public-policy-value $rules $model $log $budget)
        if($LASTEXITCODE -ne 0){throw 'Public policy/value strict replay failed'}
        $repeat=@(& $engine replay-public-policy-value $rules $model $log $budget)
        if($LASTEXITCODE -ne 0 -or ($scores -join "`n") -cne ($repeat -join "`n")){
            throw 'Public policy/value strict replay was not reproducible'
        }
        $collision=$false
        $moved=$false
        for($i=0;$i -lt $scores.Count;$i++){
            $row=ConvertFrom-Json -InputObject $scores[$i]
            $sum=($row.policy | ForEach-Object { $_[2] } | Measure-Object -Sum).Sum
            if($row.ply -ne $i -or $row.encodings -ne 1 -or [Math]::Abs($sum-1) -gt 0.00001){
                throw 'Public policy/value replay shape, normalize or encoding count failed'
            }
            $collision=$collision -or ($row.collision -ne 0)
            $moved=$moved -or ($row.moved -gt 0)
        }
        if($scores.Count -ne $budget -or !$moved -or !$collision){throw 'Public replay fixture lacks coverage'}
        $count+=$scores.Count
        [IO.File]::WriteAllLines((Join-Path $scratch ('scores-'+$seed+'.jsonl')),[string[]]$scores)
    }
    $clock.Stop()
    $overflowLog=Join-Path $scratch ('dev-'+$seeds[0]+'.jsonl')
    Assert-PublicPolicyValueRejectsOverflow $engine $rules $model $overflowLog $budget $scratch
    $evidence=[ordered]@{
        encoding='JQPV0001'; dtype='little-endian IEEE float32'; shape=@(4150,24,12,10,1)
        initialization='splitmix64 seed 41019, uniform grid [-0.2,0.2]'; status='research candidate'
        split='4 expression and 2 tactical training roots, 4 synthetic held-out perturbations, 2 generated replay sources'
        training=$training; development_seeds=$seeds; decisions_per_source=$budget; repeat=2
        replay_seconds=$clock.Elapsed.TotalSeconds
        model_sha256=(Get-FileHash $model).Hash; rules_sha256=(Get-FileHash $rules).Hash
        observation_sha256=(Get-FileHash $observation).Hash
        engine_sha256=(Get-FileHash $engine).Hash; fixture_sha256=(Get-FileHash $fixture).Hash
    }
    $evidence | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $scratch 'evidence.json') -Encoding UTF8
    Write-Output "PASS strict public replay sources=$($seeds.Count) scored=$count repeat=$count seconds=$($clock.Elapsed.TotalSeconds)"
    Write-Output "Public policy/value model_sha256=$($evidence.model_sha256) engine_sha256=$($evidence.engine_sha256)"
}

$timer=[Diagnostics.Stopwatch]::StartNew()
& (Join-Path $PSScriptRoot 'build.ps1')
Invoke-PublicPolicyValueCheck
if($PublicPolicyValueOnly){return}
& $python -B -X utf8 (Join-Path $PSScriptRoot 'verify_hypothesis_archive.py') --preflight
if($LASTEXITCODE -ne 0){throw 'Hypothesis reply history fixtures failed'}
$node=(Get-Command node -ErrorAction Stop).Source
& $python (Join-Path $PSScriptRoot 'integration.py')
if($LASTEXITCODE -ne 0){throw 'Integration failed'}
& $node (Join-Path $root 'tests\web_logic.cjs')
if($LASTEXITCODE -ne 0){throw 'Web logic failed'}
& (Join-Path $PSScriptRoot 'package.ps1')
& $python (Join-Path $PSScriptRoot 'check_engineering.py')
if($LASTEXITCODE -ne 0){throw 'Engineering check failed'}
& $python (Join-Path $root 'tests\check_guard.py')
if($LASTEXITCODE -ne 0){throw 'Negative fixture failed'}
& $python (Join-Path $root 'tests\serial_worker_test.py')
if($LASTEXITCODE -ne 0){throw 'Serial receipt fixtures failed'}
if(Test-Path -LiteralPath (Join-Path $root 'ops/task.json')){
    & $python (Join-Path $PSScriptRoot 'check_scope.py')
    if($LASTEXITCODE -ne 0){throw 'Task scope failed'}
}else{
    Write-Output 'SKIP local task scope (no local research contract in this source checkout)'
}
$timer.Stop()
Write-Output ('CHECK PASS elapsed_seconds='+$timer.Elapsed.TotalSeconds)
Write-Output 'UNVERIFIED: actual browser visual acceptance, QQ compatibility, independent topology. No inference/search/playing-strength claims.'
