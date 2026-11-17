$rows = @(Get-CimInstance Win32_Process | ForEach-Object {
  [PSCustomObject]@{
    pid = $_.ProcessId
    ppid = $_.ParentProcessId
    cl  = [string]$_.CommandLine
  }
})
($rows | ConvertTo-Json -Compress)
