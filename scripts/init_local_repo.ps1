$ErrorActionPreference = "Stop"
$repoPath = "D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model3.0"
New-Item -ItemType Directory -Force -Path $repoPath | Out-Null
Set-Location $repoPath
git init
git status
Write-Host "Copy the initialization package contents into this directory, then run:"
Write-Host "git add README.md docs config pipelines models extreme fusion runtime scripts archive .gitignore"
Write-Host "git commit -m `"Initialize 3.0 command center docs`""
