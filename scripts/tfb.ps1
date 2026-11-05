# Запуск через настоящий Python.
# Команда python в терминале Windows часто указывает на заглушку магазина приложений
# и не выполняет скрипт. Этот файл сам находит Anaconda или TFB_PYTHON.

param(
    [Parameter(Position = 0)]
    [string]$Command = "help",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Find-ProjectPython {
    $candidates = @()
    if ($env:TFB_PYTHON) { $candidates += $env:TFB_PYTHON }
    $candidates += (Join-Path $Repo ".venv\Scripts\python.exe")
    $candidates += (Join-Path $env:USERPROFILE "anaconda3\python.exe")
    $candidates += (Join-Path $env:USERPROFILE "miniconda3\python.exe")
    foreach ($path in $candidates) {
        if ($path -and (Test-Path -LiteralPath $path) -and ($path -notmatch "WindowsApps")) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    throw "Не найден Python. Задайте TFB_PYTHON или установите Anaconda в $env:USERPROFILE\anaconda3."
}

$Python = Find-ProjectPython

switch ($Command) {
    "read-panel" {
        & $Python (Join-Path $Repo "scripts\read_panel.py") @Rest
        exit $LASTEXITCODE
    }
    "baseline" {
        & $Python (Join-Path $Repo "scripts\run_baseline.py") @Rest
        exit $LASTEXITCODE
    }
    "context" {
        & $Python (Join-Path $Repo "scripts\run_context.py") @Rest
        exit $LASTEXITCODE
    }
    "queue" {
        & $Python (Join-Path $Repo "scripts\run_queue.py") @Rest
        exit $LASTEXITCODE
    }
    "check" {
        & $Python (Join-Path $Repo "tests\check_panel_reader.py")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python (Join-Path $Repo "tests\check_assembler.py")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python (Join-Path $Repo "tests\check_context.py")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python (Join-Path $Repo "tests\check_queue.py")
        exit $LASTEXITCODE
    }
    default {
        Write-Host "Python: $Python"
        Write-Host "Команды: read-panel, baseline, context, queue, check"
        Write-Host "Пример: .\scripts\tfb.ps1 read-panel --release-root tests\fixtures\mini_release --panel D12_I5_N --split validation"
        exit 0
    }
}
