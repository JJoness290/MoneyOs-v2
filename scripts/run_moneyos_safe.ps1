$ErrorActionPreference = "Stop"

$env:MONEYOS_ASSETS_ROOT = if ($env:MONEYOS_ASSETS_ROOT) { $env:MONEYOS_ASSETS_ROOT } else { 'C:\MoneyOS\assets' }
$env:MONEYOS_OUTPUT_ROOT = if ($env:MONEYOS_OUTPUT_ROOT) { $env:MONEYOS_OUTPUT_ROOT } else { 'C:\MoneyOS\work' }
$env:MONEYOS_CACHE_ROOT = if ($env:MONEYOS_CACHE_ROOT) { $env:MONEYOS_CACHE_ROOT } else { 'C:\MoneyOS\cache' }
$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { 'C:\MoneyOS\hf' }
$env:HUGGINGFACE_HUB_CACHE = if ($env:HUGGINGFACE_HUB_CACHE) { $env:HUGGINGFACE_HUB_CACHE } else { 'C:\MoneyOS\hf\hub' }
$env:TRANSFORMERS_CACHE = if ($env:TRANSFORMERS_CACHE) { $env:TRANSFORMERS_CACHE } else { $env:HUGGINGFACE_HUB_CACHE }
$env:HUGGINGFACE_HUB_DISABLE_SYMLINKS = if ($env:HUGGINGFACE_HUB_DISABLE_SYMLINKS) { $env:HUGGINGFACE_HUB_DISABLE_SYMLINKS } else { '1' }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = if ($env:HF_HUB_DISABLE_SYMLINKS_WARNING) { $env:HF_HUB_DISABLE_SYMLINKS_WARNING } else { '1' }
$env:MONEYOS_STABILITY_MODE = if ($env:MONEYOS_STABILITY_MODE) { $env:MONEYOS_STABILITY_MODE } else { '1' }
$env:MONEYOS_MAX_CONCURRENCY = if ($env:MONEYOS_MAX_CONCURRENCY) { $env:MONEYOS_MAX_CONCURRENCY } else { '1' }
$env:MONEYOS_VRAM_FRACTION = if ($env:MONEYOS_VRAM_FRACTION) { $env:MONEYOS_VRAM_FRACTION } else { '0.70' }
$env:MONEYOS_RAM_MODE = if ($env:MONEYOS_RAM_MODE) { $env:MONEYOS_RAM_MODE } else { 'low' }
$env:MONEYOS_FFMPEG_THREADS = if ($env:MONEYOS_FFMPEG_THREADS) { $env:MONEYOS_FFMPEG_THREADS } else { '1' }

$roots = @($env:MONEYOS_ASSETS_ROOT, $env:MONEYOS_OUTPUT_ROOT, $env:MONEYOS_CACHE_ROOT, $env:HF_HOME, $env:HUGGINGFACE_HUB_CACHE)
foreach ($root in $roots) {
  New-Item -ItemType Directory -Force -Path $root | Out-Null
}

Write-Host "[SAFE] Starting MoneyOS with stability defaults"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
