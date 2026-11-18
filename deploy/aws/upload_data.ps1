# Copy the data files LatticeIQ needs from this PC to the server. Run in PowerShell from the project folder:
#
#   powershell -ExecutionPolicy Bypass -File deploy\aws\upload_data.ps1 -Ip 13.54.0.1 -Key "$HOME\Downloads\LightsailDefaultKey-ap-southeast-2.pem"
#
# The files are not in the repository (licensed data and the trained model). About 150 MB.
param(
    [Parameter(Mandatory = $true)][string]$Ip,
    [Parameter(Mandatory = $true)][string]$Key,
    [string]$User = "ubuntu",
    [string]$Dir = "LatticeX"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$files = @(
    @("data\clean\prices.parquet", "data/clean/"),
    @("data\clean\fundamentals.parquet", "data/clean/"),
    @("data\clean\sec_events.parquet", "data/clean/"),
    @("data\clean\sec_events_latest.parquet", "data/clean/"),
    @("models\latticeiq_v1.npz", "models/"),
    @("data\live\range_fit.json", "data/live/"),
    @("data\live\companies.json", "data/live/")
)
# ssh refuses a key file that other Windows users could read: limit it to you
icacls $Key /inheritance:r /grant:r "$($env:USERNAME):(R)" | Out-Null
ssh -i $Key -o StrictHostKeyChecking=accept-new "$User@$Ip" "mkdir -p ~/$Dir/data/clean ~/$Dir/data/live ~/$Dir/models"
foreach ($f in $files) {
    $src = Join-Path $root $f[0]
    if (Test-Path $src) {
        Write-Host ("  {0} ({1:N0} MB)" -f $f[0], ((Get-Item $src).Length / 1MB))
        scp -q -i $Key $src "${User}@${Ip}:~/$Dir/$($f[1])"
    } else {
        Write-Host "  skipped, not on this PC: $($f[0])"
    }
}
Write-Host "Done. On the server: nano ~/$Dir/.env, then sudo systemctl start latticeiq"
