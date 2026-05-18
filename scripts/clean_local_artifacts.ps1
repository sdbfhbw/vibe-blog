$ErrorActionPreference = "Stop"

$targets = @(
    ".coverage",
    "coverage.json",
    "coverage.lcov",
    "debug.log",
    "logs",
    "backend\.env",
    "backend\.pytest_cache",
    "backend\.tmp",
    "backend\.venv",
    "backend\cache",
    "backend\data",
    "backend\logs",
    "backend\outputs",
    "backend\uploads",
    "backend\__pycache__",
    "backend\scripts\__pycache__",
    "backend\scripts\outputs",
    "frontend\coverage",
    "frontend\dist",
    "frontend\node_modules"
)

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

foreach ($target in $targets) {
    $candidate = Join-Path $workspace $target
    if (-not (Test-Path -LiteralPath $candidate)) {
        continue
    }

    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    if (-not $resolved.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside workspace: $resolved"
    }

    Remove-Item -LiteralPath $resolved -Recurse -Force
    Write-Output "Removed $target"
}
