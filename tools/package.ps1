$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$dest=Join-Path $root 'dist\junqi-mvp'
foreach($sub in @('','config','web')){New-Item -ItemType Directory -Force -Path (Join-Path $dest $sub)|Out-Null}
Copy-Item -LiteralPath (Join-Path $root 'build\junqi.exe') -Destination (Join-Path $dest 'junqi.exe')
foreach($f in @('ruleset.json','ruleset-selfplay.json','observation.json')){Copy-Item -LiteralPath (Join-Path $root ('config\'+$f)) -Destination (Join-Path $dest ('config\'+$f))}
foreach($f in @('index.html','style.css','review.js','app.js','demo.js')){Copy-Item -LiteralPath (Join-Path $root ('web\'+$f)) -Destination (Join-Path $dest ('web\'+$f))}
foreach($f in @('Start-MVP.cmd','start-mvp.ps1','README.md')){Copy-Item -LiteralPath (Join-Path $root $f) -Destination (Join-Path $dest $f)}
$bytes=(Get-ChildItem -LiteralPath $dest -Recurse -File|Measure-Object Length -Sum).Sum
Write-Output ('Package bytes='+$bytes+'; hard limit=100000000; remaining='+(100000000-$bytes))
if($bytes -gt 100000000){throw 'Package exceeds hard budget'}
