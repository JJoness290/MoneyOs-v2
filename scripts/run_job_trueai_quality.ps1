param(
  [string]$Prompt = "anime action sequence in a futuristic city"
)

$ErrorActionPreference = "Stop"
$payload = @{ prompt = $Prompt } | ConvertTo-Json
$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/jobs/anime-trueai-quality" -ContentType "application/json" -Body $payload
Write-Host "job_id=$($response.job_id)"
Write-Host "output_dir=$($response.output_dir)"
