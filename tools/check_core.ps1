$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (!(Test-Path -LiteralPath $vcvars)) { throw 'Existing MSVC toolchain not found; no installation attempted.' }
$buildRoot = Join-Path $projectRoot 'build'
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
Push-Location $buildRoot
try {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $command = 'call "' + $vcvars + '" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /I..\core ..\core\board.cpp ..\core\rules.cpp ..\core\state.cpp ..\core\layout.cpp ..\tests\core_smoke.cpp /Fe:core_smoke.exe'
    & $env:ComSpec /d /s /c $command
    if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
    & .\core_smoke.exe
    if ($LASTEXITCODE -ne 0) { throw 'Core tests failed' }
    $timer.Stop()
    Write-Output ('Build and test elapsed_seconds=' + $timer.Elapsed.TotalSeconds)
    Write-Output ('Test executable bytes=' + (Get-Item core_smoke.exe).Length)
    Write-Output 'NOT VERIFIED: QQ compatibility, GwanKei diff, Observation isolation, replay golden logs, full engineering check, release size, peak memory.'
} finally { Pop-Location }
