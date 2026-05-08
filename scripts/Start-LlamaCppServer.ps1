param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [ValidateSet("auto", "on", "off")]
    [string]$Gpu = "auto",

    [int]$GpuLayers = -1,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$ContextSize = 4096,
    [int]$Threads = 8,
    [string]$LogPath = "out/local-setup/logs/llama-server.log",
    [switch]$Background,
    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-HostNvidiaGpu {
    if (-not (Test-CommandExists "nvidia-smi")) {
        return $false
    }

    $null = & nvidia-smi -L 2>$null
    return $LASTEXITCODE -eq 0
}

function Join-CommandLine {
    param([string[]]$Parts)
    return ($Parts | ForEach-Object {
        if ($_ -match "\s") {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join " "
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$resolvedModel = Resolve-Path $ModelPath
$resolvedLog = Join-Path $repoRoot $LogPath
$logDir = Split-Path -Parent $resolvedLog

$gpuSelected = $false
$gpuReason = "disabled"
if ($Gpu -eq "on") {
    $gpuSelected = $true
    $gpuReason = "requested"
} elseif ($Gpu -eq "auto") {
    if (Test-HostNvidiaGpu) {
        $gpuSelected = $true
        $gpuReason = "nvidia-smi detected"
    } else {
        $gpuReason = "nvidia-smi unavailable"
    }
}

$arguments = @(
    "-m", "llama_cpp.server",
    "--model", $resolvedModel.Path,
    "--host", $HostAddress,
    "--port", [string]$Port,
    "--n_ctx", [string]$ContextSize,
    "--n_threads", [string]$Threads
)

if ($gpuSelected) {
    $arguments += @("--n_gpu_layers", [string]$GpuLayers)
}

if ($ExtraArgs) {
    $arguments += $ExtraArgs
}

$command = @("python") + $arguments

Write-Host "llama.cpp GPU offload: $(if ($gpuSelected) { 'enabled' } else { 'disabled' }) ($gpuReason)"
Write-Host "Endpoint: http://$HostAddress`:$Port/v1"
Write-Host "Command: $(Join-CommandLine $command)"

if ($DryRun) {
    exit 0
}

if ($Background) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $errorLog = [System.IO.Path]::Combine(
        $logDir,
        [System.IO.Path]::GetFileNameWithoutExtension($resolvedLog) + ".err.log"
    )
    $process = Start-Process `
        -FilePath "python" `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $resolvedLog `
        -RedirectStandardError $errorLog `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "Started llama.cpp server PID $($process.Id); stdout: $resolvedLog; stderr: $errorLog"
    exit 0
}

Push-Location $repoRoot
try {
    & python @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
