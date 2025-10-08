# Minimal PowerShell watcher for renaming new .txt files.
param([string]$WatchDir = "D:\BLE_Output")
$fsw = New-Object System.IO.FileSystemWatcher
$fsw.Path = $WatchDir
$fsw.Filter = "*.txt"
$fsw.IncludeSubdirectories = $false
$fsw.EnableRaisingEvents = $true
$action = {
  $path = $Event.SourceEventArgs.FullPath
  $prev = -1; $stable = 0
  while ($true) {
    if (-not (Test-Path $path)) { break }
    try { $size = (Get-Item $path).Length } catch { $size = -1 }
    if ($size -eq $prev -and $size -ge 0) { $stable += 1 } else { $stable = 0 }
    $prev = $size
    if ($stable -ge 2) { break }
    Start-Sleep -Milliseconds 500
  }
  if (-not (Test-Path $path)) { return }
  $fi = Get-Item $path
  $ts = $fi.LastWriteTime
  $date = $ts.ToString("yyyyMMdd")
  $time = $ts.ToString("HHmmss")
  $device = "NODEV"
  try {
    $content = Get-Content -Path $path -Raw -ErrorAction SilentlyContinue
    $m = [regex]::Match($content, '(?i)(device|mac)\s*[:=]\s*([0-9A-F]{2}([:\-][0-9A-F]{2}){5})')
    if ($m.Success) { $device = $m.Groups[2].Value.Replace(":", "-") }
  } catch {}
  $destDir = Join-Path $WatchDir ($ts.ToString("yyyy-MM-dd"))
  if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
  $seq = (Get-ChildItem -Path $destDir -Filter "*$date*").Count + 1
  $newName = "{0}-{1}-{2}-{3:D3}.txt" -f $date, $time, $device, $seq
  $destPath = Join-Path $destDir $newName
  try {
    $header = "# source_file: {0}`r`n# timestamp: {1}`r`n" -f $fi.Name, ($ts.ToString("yyyy-MM-dd HH:mm:ss"))
    $body = Get-Content -Path $path -Raw
    $normalized = $body -replace "(`r`n|`n|`r)", "`r`n"
    $final = $header + "`r`n" + $normalized
    Set-Content -Path $destPath -Value $final -Encoding UTF8
    Rename-Item -Path $path -NewName ($fi.Name + ".bak") -ErrorAction SilentlyContinue
  } catch {}
}
$created = Register-ObjectEvent $fsw Created -Action $action
while ($true) { Start-Sleep -Seconds 1 }