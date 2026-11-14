param([string]$Destination = (Join-Path $PSScriptRoot '../models/u2netp.onnx'))
$ErrorActionPreference = 'Stop'
$modelDestination = [System.IO.Path]::GetFullPath($Destination)
$modelDirectory = [System.IO.Path]::GetDirectoryName($modelDestination)
New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
$modelTemporary = "$modelDestination.download"
Invoke-WebRequest -Uri 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx' -OutFile $modelTemporary
$modelChecksum = (Get-FileHash -LiteralPath $modelTemporary -Algorithm MD5).Hash.ToLowerInvariant()
if ($modelChecksum -ne '8e83ca70e441ab06c318d82300c84806') { throw 'U2NetP checksum mismatch; downloaded file was not installed.' }
Move-Item -LiteralPath $modelTemporary -Destination $modelDestination -Force
Write-Output "Installed verified U2NetP model: $modelDestination"
Get-FileHash -LiteralPath $modelDestination -Algorithm SHA256
