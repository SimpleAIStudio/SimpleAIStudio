$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================"
Write-Host "              SIMPLEAI INSTALL"
Write-Host "============================================"
Write-Host ""

$CliFolder = Split-Path -Parent $MyInvocation.MyCommand.Path

$ProjectRoot = Split-Path -Parent $CliFolder

$SourcePython = Join-Path $CliFolder "simpleai.py"

$SourceRegistry = Join-Path $ProjectRoot "registry\models.json"

$InstallFolder = Join-Path $env:LOCALAPPDATA "SimpleAI"

$RegistryFolder = Join-Path $InstallFolder "registry"

$InstalledPython = Join-Path $InstallFolder "simpleai.py"

$CommandFile = Join-Path $InstallFolder "simpleai.cmd"


if (-not (Test-Path $SourcePython)) {

    Write-Host "ERROR: simpleai.py was not found."
    exit 1

}


if (-not (Test-Path $SourceRegistry)) {

    Write-Host "ERROR: registry\models.json was not found."
    exit 1

}


Write-Host "Checking Python..."

$PythonTest = Get-Command py -ErrorAction SilentlyContinue

if (-not $PythonTest) {

    Write-Host ""
    Write-Host "ERROR: Python launcher 'py' was not found."
    Write-Host ""
    Write-Host "Install Python first, then run this installer again."
    Write-Host ""

    exit 1

}


Write-Host "Python found."
Write-Host ""

Write-Host "Creating SimpleAI installation..."

New-Item `
    -ItemType Directory `
    -Path $InstallFolder `
    -Force `
    | Out-Null

New-Item `
    -ItemType Directory `
    -Path $RegistryFolder `
    -Force `
    | Out-Null


Copy-Item `
    $SourcePython `
    $InstalledPython `
    -Force


Copy-Item `
    $SourceRegistry `
    (Join-Path $RegistryFolder "models.json") `
    -Force


$CommandContents = @'
@echo off
py "%LOCALAPPDATA%\SimpleAI\simpleai.py" %*
'@


Set-Content `
    -Path $CommandFile `
    -Value $CommandContents `
    -Encoding ASCII


Write-Host "CLI files installed."
Write-Host ""

Write-Host "Checking PATH..."


$UserPath = [Environment]::GetEnvironmentVariable(
    "Path",
    "User"
)


if ([string]::IsNullOrWhiteSpace($UserPath)) {

    $UserPath = ""

}


$PathEntries = $UserPath -split ";"


$AlreadyExists = $false


foreach ($Entry in $PathEntries) {

    if (
        $Entry.TrimEnd("\") -ieq
        $InstallFolder.TrimEnd("\")
    ) {

        $AlreadyExists = $true
        break

    }

}


if (-not $AlreadyExists) {

    if (
        $UserPath.Length -gt 0 -and
        -not $UserPath.EndsWith(";")
    ) {

        $UserPath += ";"

    }


    $UserPath += $InstallFolder


    [Environment]::SetEnvironmentVariable(
        "Path",
        $UserPath,
        "User"
    )


    Write-Host "Added SimpleAI to your user PATH."

}
else {

    Write-Host "SimpleAI is already in your PATH."

}


if (
    -not (
        ($env:Path -split ";") |
        Where-Object {
            $_.TrimEnd("\") -ieq
            $InstallFolder.TrimEnd("\")
        }
    )
) {

    $env:Path += ";$InstallFolder"

}


$ModelFolder = Join-Path $HOME ".simpleai\models"

New-Item `
    -ItemType Directory `
    -Path $ModelFolder `
    -Force `
    | Out-Null


Write-Host ""
Write-Host "           SIMPLEAI INSTALLED"
Write-Host ""
Write-Host "Installed to:"
Write-Host "  $InstallFolder"
Write-Host ""
Write-Host "Models will live in:"
Write-Host "  $ModelFolder"
Write-Host ""
Write-Host "Try:"
Write-Host ""
Write-Host "  simpleai version"
Write-Host "  simpleai list"
Write-Host "  simpleai info nano-dumbthink:1.0"
Write-Host ""