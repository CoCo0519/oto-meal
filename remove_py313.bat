# 0) 清掉可能存在的别名/函数覆盖（当前会话）
# 0) Clear possible alias/function hijacks (current session)
Remove-Item Alias:python   -ErrorAction SilentlyContinue
Remove-Item Function:python -ErrorAction SilentlyContinue

# 1) 在用户目录创建一个“shims”文件夹
# 1) Create a per-user shims folder
$shim = "$HOME\ps-shims"
New-Item -ItemType Directory -Force $shim | Out-Null

# 2) 写一个 python.cmd，把所有调用转发到 3.11
#    这里用 Windows Python Launcher 保证指向 3.11（你 py -V 已是 3.11）
# 2) Write python.cmd to forward to 3.11 via the Windows Python Launcher
Set-Content -Encoding ASCII -Path "$shim\python.cmd" -Value "@echo off`r`npy -3.11 %*"

# （可选）pip 也一起转发到 3.11 环境
# (Optional) forward pip to 3.11 too
Set-Content -Encoding ASCII -Path "$shim\pip.cmd" -Value "@echo off`r`npy -3.11 -m pip %*"

# 3) 把这个 shims 目录放到“用户级 PATH”的最前面（无需管理员）
# 3) Prepend the shims folder to USER PATH (no admin required)
$userPath = [Environment]::GetEnvironmentVariable('Path','User')
if (($userPath -split ';') -notcontains $shim) {
  [Environment]::SetEnvironmentVariable('Path', "$shim;$userPath", 'User')
}

# 4) 关闭 Windows 的“应用执行别名”（避免商店别名抢占）
# 4) Make sure App Execution Aliases are off for python/python3 (manual UI step)
Write-Host "请到：设置 → 应用 → 应用执行别名，关闭 python.exe / python3.exe（如果开启）。"

# 5) 自我重启当前 PowerShell 以加载新的 PATH
# 5) Restart this PowerShell to load updated PATH
Start-Process -FilePath (Get-Process -Id $PID).Path -WorkingDirectory $PWD; exit
