$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$vcvars='C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Force -Path (Join-Path $root 'build') | Out-Null
Push-Location (Join-Path $root 'build')
try {
  $common='/nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 /I..\core ..\core\board.cpp ..\core\rules.cpp ..\core\state.cpp ..\core\layout.cpp '
  foreach($item in @(@('..\tests\core_smoke.cpp','core_smoke.exe'),@('..\tests\mvp_core.cpp','mvp_core.exe'),@('..\sim\main.cpp','junqi.exe'))) {
    & $env:ComSpec /d /s /c ('call "'+$vcvars+'" >nul && cl '+$common+$item[0]+' /Fe:'+$item[1])
    if($LASTEXITCODE -ne 0){throw 'Build failed'}
  }
  foreach($header in Get-ChildItem -Path ($root+'\core'),($root+'\observe'),($root+'\strategy'),($root+'\sim') -Filter *.h -Recurse) {
    $include=$header.FullName.Replace('\','/')
    Set-Content -Path header_probe.cpp -Value ('#include "'+$include+'"') -Encoding utf8
    & $env:ComSpec /d /s /c ('call "'+$vcvars+'" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /utf-8 /Zs header_probe.cpp')
    if($LASTEXITCODE -ne 0){throw ('Header is not self-contained: '+$include)}
  }
  & .\core_smoke.exe
  if($LASTEXITCODE -ne 0){throw 'Smoke failed'}
  & .\mvp_core.exe
  if($LASTEXITCODE -ne 0){throw 'MVP tests failed'}
} finally {Pop-Location}
