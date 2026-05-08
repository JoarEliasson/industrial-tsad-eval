param(
    [ValidateSet("auto", "on", "off")]
    [string]$Gpu = "auto",

    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ComposeArgs
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

function Test-DockerCudaReady {
    $null = & docker image inspect industrial-tsad-eval:local 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    $probe = & docker run --rm --gpus all industrial-tsad-eval:local `
        python -c "import torch; print('cuda=' + str(torch.cuda.is_available()).lower())" 2>$null
    return $LASTEXITCODE -eq 0 -and ($probe -match "cuda=true")
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

if (-not $ComposeArgs -or $ComposeArgs.Count -eq 0) {
    $ComposeArgs = @("run", "--rm", "itse", "itse", "--help")
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$composeFiles = @("-f", (Join-Path $repoRoot "docker-compose.yml"))
$gpuSelected = $false
$gpuReason = "disabled"

if ($Gpu -eq "on") {
    $gpuSelected = $true
    $gpuReason = "requested"
} elseif ($Gpu -eq "auto") {
    if ((Test-HostNvidiaGpu) -and (Test-DockerCudaReady)) {
        $gpuSelected = $true
        $gpuReason = "auto-detected"
    } else {
        $gpuReason = "not available"
    }
}

if ($gpuSelected) {
    $composeFiles += @("-f", (Join-Path $repoRoot "docker-compose.gpu.yml"))
}

$command = @("docker", "compose") + $composeFiles + $ComposeArgs

Write-Host "Docker GPU route: $(if ($gpuSelected) { 'enabled' } else { 'disabled' }) ($gpuReason)"
Write-Host "Command: $(Join-CommandLine $command)"

if ($DryRun) {
    exit 0
}

Push-Location $repoRoot
try {
    & docker compose @composeFiles @ComposeArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
