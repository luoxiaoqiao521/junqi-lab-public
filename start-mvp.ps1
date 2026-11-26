param([UInt64]$Seed=42,[switch]$NoOpen,[string]$RunDirectory,
 [ValidateSet('flag-route','random','heuristic-ns','heuristic-ew','heuristic-all','heuristic-south','heuristic-east',
 'heuristic-2-ns','heuristic-2-ew','heuristic-2-all','heuristic-2-south','heuristic-2-east',
 'heuristic-3-ns','heuristic-3-ew','heuristic-3-all','heuristic-3-south','heuristic-3-east')][string]$Policy='random',
 [ValidateSet('four-dark','single-dark','four-dual','four-full')][string]$Mode='four-dark',[switch]$ArchiveFlagReveal,[switch]$ArchiveQQ,[string]$ArchiveFile)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
if(!$PSBoundParameters.ContainsKey('Policy') -and !$ArchiveFile -and !$ArchiveFlagReveal -and !$ArchiveQQ){$Policy='flag-route'}
$practice=$Policy -eq 'flag-route'
if($practice -and ($ArchiveQQ -or $ArchiveFlagReveal)){throw 'flag-route uses the local practice rule profile'}
if($ArchiveQQ -and $Mode -eq 'single-dark'){throw 'ArchiveQQ requires a four-seat mode'}
if($ArchiveFile){
 if($Mode -ne 'four-dark' -or $ArchiveQQ -or $ArchiveFlagReveal -or $Policy -ne 'random'){throw 'ArchiveFile uses the strict archival profile'}
 $ArchiveFile=(Resolve-Path -LiteralPath $ArchiveFile).ProviderPath
}
$exe=Join-Path $root 'junqi.exe'
if(!(Test-Path -LiteralPath $exe)){$exe=Join-Path $root 'build\junqi.exe'}
if(!(Test-Path -LiteralPath $exe)){throw 'Build junqi.exe first with tools/build.ps1'}
$userRoot=if($RunDirectory){[IO.Path]::GetFullPath($RunDirectory)}else{
 Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'JunqiLab\runs'
}
New-Item -ItemType Directory -Force -Path $userRoot|Out-Null
$run=Join-Path $userRoot ((Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+[Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $run|Out-Null
$log=Join-Path $run 'game.jsonl'
$command=switch($Mode){
 'single-dark'{'simulate-single'}
 'four-dual'{'simulate-dual'}
 'four-full'{'simulate-full'}
 default{'simulate'}
}
$rulesPath=Join-Path $root $(if($practice){'config\ruleset-selfplay.json'}else{'config\ruleset.json'})
$observationPath=Join-Path $root 'config\observation.json'
if($practice -or $ArchiveFlagReveal -or $ArchiveQQ){
 $rules=Get-Content -LiteralPath $rulesPath -Raw|ConvertFrom-Json
 $observation=Get-Content -LiteralPath $observationPath -Raw|ConvertFrom-Json
 $rules.experimental.'marshal_death.reveal_flag'=$true
 $rules.evidence.'marshal_death.reveal_flag'=if($practice){'user_defined_offline_practice'}else{'observed_in_teaonly_qq_archive'}
 $observation.marshal_reveals_flag=$true
 if($practice){
  $rules.id='flag-practice-1'
  $observation.id='four-dark-flag-practice-1'
 }
 if($ArchiveQQ){
  $rules.id='qq-four-archive-experimental-1'
  $rules.experimental.'turn.order'='cw'
  $rules.experimental.'elim.no_legal_move'='eliminated_after_involved_move'
  $rules.experimental.'draw.no_progress.limit'=71
  $rules.evidence.'turn.order'='most_adjacent_turns_in_qq_archive_coordinates'
  $rules.evidence.'elim.no_legal_move'='archive_best_fit_with_five_exceptions'
  $rules.evidence.'draw.no_progress.limit'='observed_71_quiet_plies_in_qq_archives'
 }
 $rulesPath=Join-Path $run 'ruleset.json'
 $observationPath=Join-Path $run 'observation.json'
 $encoding=[Text.UTF8Encoding]::new($false)
 [IO.File]::WriteAllText($rulesPath,($rules|ConvertTo-Json -Depth 8),$encoding)
 [IO.File]::WriteAllText($observationPath,($observation|ConvertTo-Json -Depth 8),$encoding)
}
if($ArchiveFile){
 & $exe import-jgs $ArchiveFile $log $rulesPath $observationPath
 if($LASTEXITCODE -ne 0){throw 'JGS replay validation failed'}
 & $exe export $log (Join-Path $run 'demo.js')
}else{
 & $exe $command $Seed $log $rulesPath $observationPath $Policy
 if($LASTEXITCODE -ne 0){throw 'Simulation failed'}
 & $exe export-review $log (Join-Path $run 'demo.js')
}
if($LASTEXITCODE -ne 0){throw 'Replay verification/export failed'}
foreach($f in @('index.html','style.css','review.js','app.js')){Copy-Item -LiteralPath (Join-Path $root ('web\'+$f)) -Destination (Join-Path $run $f)}
Write-Output ('Replay: '+(Join-Path $run 'index.html'))
Write-Output ('Private log: '+$log)
$count=(Get-ChildItem -LiteralPath $userRoot -Directory|Measure-Object).Count
if($count -gt 30){Write-Warning 'More than 30 saved runs. Archive old runs manually; nothing was deleted.'}
if(!$NoOpen){Start-Process -FilePath (Join-Path $run 'index.html')}
