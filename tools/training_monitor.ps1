param(
  [Parameter(Mandatory)][string]$Directory,
  [ValidateRange(1,2147483647)][int]$Hours = 6,
  [switch]$Snapshot
)
$ErrorActionPreference = 'Stop'
$monitorStarted = [DateTime]::UtcNow
$runDirectory = [IO.Path]::GetFullPath($Directory)
$buildDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../build'))
if (!$runDirectory.StartsWith($buildDirectory + [IO.Path]::DirectorySeparatorChar)) {
  throw 'Monitor target must be a run directory inside project build'
}

function Read-TrainingStatus {
  $files = @(Get-ChildItem -LiteralPath $runDirectory -File -ErrorAction SilentlyContinue)
  $rounds = @($files | Where-Object Name -Match '^round-\d+-\d+\.json$')
  $bytes = ($files | Measure-Object Length -Sum).Sum
  $summaryPath = Join-Path $runDirectory 'summary.json'
  $summary = $null
  if (Test-Path -LiteralPath $summaryPath) {
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
  }
  $progressPath = Join-Path $runDirectory 'progress.json'
  $progress = if (Test-Path -LiteralPath $progressPath) {
    Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
  }
  $profile = if ($progress.profile) {$progress.profile} else {$summary.profile}
  $archive = $profile -eq 'archive-pressure'
  $practice = $profile -eq 'archive-pressure-practice'
  $current = if (($archive -or $practice) -and $progress) {$progress} else {$summary}
  $status = if ($current) {$current.status} else {'running'}
  if ($archive) {$rounds = @($files | Where-Object Name -Match '^(ordinary|pressure)\.json$')}
  if ($practice) {$rounds = @($files | Where-Object Name -eq 'summary.json')}
  $latest = $rounds | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
  $manifest = $files | Where-Object Name -EQ 'manifest.json' | Select-Object -First 1
  $total = 16
  if ($manifest) {
    $configuration = Get-Content -LiteralPath $manifest.FullName -Raw | ConvertFrom-Json
    if ($configuration.total_batches) {$total = [int]$configuration.total_batches}
  }
  if ($archive) {$total = 2}
  if ($practice) {$total = 1}
  $eta = if ($status -eq 'complete') {0} else {$null}
  if ($latest -and $status -eq 'running' -and !$archive -and !$practice) {
    $lastRound = Get-Content -LiteralPath $latest.FullName -Raw | ConvertFrom-Json
    $eta = [Math]::Ceiling(($total - $rounds.Count) * $lastRound.seconds)
  }
  $manifest = $files | Where-Object Name -EQ 'manifest.json' | Select-Object -First 1
  $began = if ($manifest) {$manifest.LastWriteTimeUtc} else {$monitorStarted}
  if ($current -and $current.started_utc) {
    $began = [DateTimeOffset]::FromUnixTimeSeconds([long]$current.started_utc).UtcDateTime
  }
  $limit = if ($current.hours_limit) {$current.hours_limit} else {$Hours}
  [pscustomobject]@{
    Status = $status
    Phase = $current.phase
    Completed = $rounds.Count
    Total = $total
    ArtifactBytes = [long]$bytes
    EstimatedSeconds = $eta
    HardLimitSeconds = [Math]::Max(0, [Math]::Ceiling(($began.AddHours($limit) - [DateTime]::UtcNow).TotalSeconds))
    QuietSeconds = if ($latest) {([DateTime]::UtcNow - $latest.LastWriteTimeUtc).TotalSeconds} else {([DateTime]::UtcNow - $began).TotalSeconds}
    Error = if ($current) {$current.error} else {$null}
  }
}

if ($Snapshot) {
  Read-TrainingStatus | ConvertTo-Json
  exit 0
}
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$identity = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($runDirectory)))
$created = $false
$mutex = [Threading.Mutex]::new($true, "Local\JunqiMonitor-$identity", [ref]$created)
if (!$created) {$mutex.Dispose(); exit 0}
$tray = [Windows.Forms.NotifyIcon]::new()
$tray.Icon = [Drawing.SystemIcons]::Information
$tray.Text = '军棋训练监控'
$tray.Visible = $true
$menu = [Windows.Forms.ContextMenuStrip]::new()
$statusItem = $menu.Items.Add('正在读取进度')
$statusItem.Enabled = $false
$remainingItem = $menu.Items.Add('预计剩余时间待测')
$remainingItem.Enabled = $false
$limitItem = $menu.Items.Add('硬截止时间待测')
$limitItem.Enabled = $false
$bytesItem = $menu.Items.Add('正在读取产物大小')
$bytesItem.Enabled = $false
$openItem = $menu.Items.Add('打开训练结果目录')
$openItem.Add_Click({Start-Process explorer.exe -ArgumentList $runDirectory})
$stopItem = $menu.Items.Add('本阶段完成后停止训练')
$stopItem.Add_Click({
  $path = Join-Path $runDirectory 'STOP'
  if (!(Test-Path -LiteralPath $path)) {[IO.File]::WriteAllText($path,'user requested stop')}
})
$exitItem = $menu.Items.Add('关闭监控')
$exitItem.Add_Click({[Windows.Forms.Application]::ExitThread()})
$tray.ContextMenuStrip = $menu
$timer = [Windows.Forms.Timer]::new()
$timer.Interval = 10000
$script:notice = ''
$script:finishedAt = $null
$script:readFailures = 0
$script:shownStatus = $null

function Show-TrainingStatus {
  try {
    $state = Read-TrainingStatus
    $script:readFailures = 0
    $names = @{running='训练中'; complete='已完成'; failed='出错'; stopped='已停止'; storage_limit='达到存储上限'; insufficient_coverage='样本覆盖不足'}
    $phases = @{locate='定位棋谱'; scan='核验前缀'; train='训练两组'; assess='留出评测'; practice='旗口配对练习'}
    $label = if ($names.ContainsKey($state.Status)) {$names[$state.Status]} else {$state.Status}
    if ($state.Status -eq 'running' -and $state.Phase -and $phases.ContainsKey($state.Phase)) {
      $label = $phases[$state.Phase]
    }
    $statusItem.Text = "$label · $($state.Completed)/$($state.Total) 批"
    $remainingItem.Text = if ($state.Status -eq 'complete') {'预计剩余 0 秒'} elseif ($null -eq $state.EstimatedSeconds) {'预计剩余时间待首批完成'} else {"预计剩余约 $($state.EstimatedSeconds) 秒"}
    $limitItem.Text = "硬时限剩余 $([Math]::Round($state.HardLimitSeconds/3600,2)) 小时"
    $bytesItem.Text = "本次产物 $([Math]::Round($state.ArtifactBytes/1MB,2)) MiB / 32 MiB"
    $tray.Text = "军棋 $($state.Completed)/$($state.Total) · $label · 预计$($state.EstimatedSeconds)秒"
    $stopItem.Enabled = $state.Status -eq 'running'
    $warning = ''
    if ($state.Status -eq 'running' -and $state.ArtifactBytes -gt 32MB) {
      $warning = '产物超出32MiB上限，已请求阶段边界停止'
    } elseif ($state.Status -eq 'running' -and $state.QuietSeconds -gt 600) {
      $warning = '十分钟无新检查点，已请求阶段边界停止'
    }
    if ($warning) {
      $stopPath = Join-Path $runDirectory 'STOP'
      if (!(Test-Path -LiteralPath $stopPath)) {[IO.File]::WriteAllText($stopPath,$warning)}
    }
    $message = if ($warning) {$warning} elseif ($state.Status -eq 'failed') {"训练出错，详情见运行目录。$($state.Error)"} else {"$($statusItem.Text)。$($remainingItem.Text)。$($bytesItem.Text)"}
    $key = if ($warning) {$warning} else {$state.Status}
    if ($key -ne $script:notice) {
      $tray.BalloonTipTitle = '军棋训练'
      $tray.BalloonTipText = $message.Substring(0,[Math]::Min(220,$message.Length))
      $tray.ShowBalloonTip(7000)
      $script:notice = $key
    }
    if ($state.Status -ne 'running' -and !$script:finishedAt) {$script:finishedAt = [DateTime]::UtcNow}
    if ($script:finishedAt -and ([DateTime]::UtcNow - $script:finishedAt).TotalMinutes -gt 15) {
      [Windows.Forms.Application]::ExitThread()
    }
    $script:shownStatus = $state
  } catch {
    $script:readFailures++
    if ($script:readFailures -eq 3) {
      $tray.BalloonTipTitle = '军棋监控读取失败'
      $tray.BalloonTipText = '连续三次读取失败，检查运行目录。监控不会重新启动训练。'
      $tray.ShowBalloonTip(7000)
    }
  }
}
$tray.Add_DoubleClick({
  if ($script:shownStatus) {
    $message = @($statusItem.Text,$remainingItem.Text,$limitItem.Text,$bytesItem.Text) -join [Environment]::NewLine
    [Windows.Forms.MessageBox]::Show($message, '军棋训练进度') | Out-Null
  }
})
$timer.Add_Tick({Show-TrainingStatus})
try {
  Show-TrainingStatus
  $timer.Start()
  [Windows.Forms.Application]::Run()
} finally {
  $timer.Stop()
  $timer.Dispose()
  $tray.Visible = $false
  $tray.Dispose()
  $menu.Dispose()
  $mutex.ReleaseMutex()
  $mutex.Dispose()
}
