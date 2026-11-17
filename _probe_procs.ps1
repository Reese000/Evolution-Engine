Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object ProcessId, CreationDate, @{n='Cmd';e={$_.CommandLine}} |
  Format-List
