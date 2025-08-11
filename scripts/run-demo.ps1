param(
  [switch]$Release
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# repo root (this file lives in scripts/)
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

# check models exist
$enc = Join-Path $root "models\sam_encoder.onnx"
$dec = Join-Path $root "models\sam_decoder.onnx"
if (-not (Test-Path $enc)) { throw "[ERROR] Missing models\sam_encoder.onnx" }
if (-not (Test-Path $dec)) { throw "[ERROR] Missing models\sam_decoder.onnx" }

# optional: sample image hint (not required to exist here, csproj copies from assets)
$img = Join-Path $root "assets\ImageSample.jpg"
if (-not (Test-Path $img)) { Write-Warning "[WARN] assets\ImageSample.jpg not found (demo will still run if your code reads another image)" }

# restore & run
$projDir = Join-Path $root "SamSharp_with_Demo"
Set-Location $projDir
& dotnet restore
$cfg = if ($Release) { "Release" } else { "Debug" }
& dotnet run -c $cfg
