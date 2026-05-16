param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("hydrate", "export", "clear")]
    [string]$Action,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

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
$composeFiles = @(
    "-f", (Join-Path $repoRoot "docker-compose.yml"),
    "-f", (Join-Path $repoRoot "docker-compose.fast-io.yml")
)

switch ($Action) {
    "hydrate" {
        $script = @"
set -eu
mkdir -p /workspace/prepared
rm -rf /workspace/prepared/*
cp -a /host/prepared/. /workspace/prepared/
find /workspace/prepared -maxdepth 2 -type d | wc -l
"@
    }
    "export" {
        $script = @"
set -eu
mkdir -p /host/out
cp -a /workspace/out/. /host/out/
find /host/out -maxdepth 2 -type d | wc -l
"@
    }
    "clear" {
        $script = @"
set -eu
rm -rf /workspace/prepared/* /workspace/out/*
echo cleared-fast-io-volumes
"@
    }
}

$command = @("docker", "compose") + $composeFiles + @(
    "run", "--rm", "--entrypoint", "sh", "itse", "-lc", $script
)

Write-Host "Fast-I/O action: $Action"
Write-Host "Command: $(Join-CommandLine $command)"

if ($DryRun) {
    exit 0
}

Push-Location $repoRoot
try {
    & docker compose @composeFiles run --rm --entrypoint sh itse -lc $script
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
