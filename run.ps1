Clear-Host
$CarlaPath = "C:\Users\User\Documents\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe"
$ConeSpawnJson = "$PSScriptRoot\env\cone_spawn.json"

$run_name = Read-Host "Enter run name"

# --- Scenario selection ---
# Parse valid scenario names from cone_spawn.json (_scenario tags)
$coneData = Get-Content $ConeSpawnJson -Raw | ConvertFrom-Json
$scenarioNames = @("All Scenarios")  # index 0 = no filter
foreach ($cone in $coneData) {
    if ($cone.PSObject.Properties.Name -contains "_scenario") {
        $scenarioNames += $cone._scenario
    }
}

Write-Host "`n=== Select Training Scenario ===" -ForegroundColor Cyan
for ($i = 0; $i -lt $scenarioNames.Count; $i++) {
    Write-Host "$($i + 1)) $($scenarioNames[$i])" -ForegroundColor Yellow
}

$scenarioSel = Read-Host "`nSelect scenario [1-$($scenarioNames.Count)] (default: 1)"
if ([string]::IsNullOrWhiteSpace($scenarioSel)) { $scenarioSel = "1" }
$scenarioIdx = [int]$scenarioSel - 1

if ($scenarioIdx -lt 0 -or $scenarioIdx -ge $scenarioNames.Count) {
    Write-Host "Invalid scenario selection. Defaulting to All Scenarios." -ForegroundColor Red
    $scenarioIdx = 0
}

$selectedScenario = $scenarioNames[$scenarioIdx]
Write-Host "Selected: $selectedScenario" -ForegroundColor Green

# Build --scenario argument (empty string means all scenarios)
$scenarioArg = ""
if ($scenarioIdx -ne 0) {
    $scenarioArg = "--scenario `"$selectedScenario`""
}

# Helper function to check if a port is actively listening
function Test-PortActive($Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

Write-Host "`n=== CARLA Automation Menu ===" -ForegroundColor Cyan
Write-Host "1) Start Single Server & Train" -ForegroundColor Yellow
Write-Host "2) Start Parallel Servers & Train" -ForegroundColor Yellow
Write-Host "3) Evaluate Best Model" -ForegroundColor Yellow
Write-Host "4) Exit" -ForegroundColor Yellow

$selection = Read-Host "`nSelect a task to run [1-4] (default: 1)"
if ([string]::IsNullOrWhiteSpace($selection)) { $selection = "1" }

switch ($selection) {
    "1" {
        Write-Host "`nChecking Port 2000..." -ForegroundColor Yellow
        if (Test-PortActive 2000) {
            Write-Host "CARLA Server is already active on Port 2000! Skipping launch..." -ForegroundColor Green
        } else {
            Write-Host "Port 2000 is free. Launching CARLA Server..." -ForegroundColor Green
            Start-Process -FilePath $CarlaPath -ArgumentList "-carla-rpc-port=2000"
            Start-Sleep -Seconds 10
        }
        Invoke-Expression "python train_cone_avoidance.py --ports 2000 $scenarioArg --run-name $run_name"
    }
    
    "2" {
        Write-Host "`nChecking Ports 2000 and 3000..." -ForegroundColor Yellow
        $launchNeeded = $false

        # Check Port 2000
        if (Test-PortActive 2000) {
            Write-Host "CARLA Server is already active on Port 2000." -ForegroundColor DarkGreen
        } else {
            Write-Host "Launching CARLA Server on Port 2000..." -ForegroundColor Green
            Start-Process -FilePath $CarlaPath -ArgumentList "-carla-rpc-port=2000"
            $launchNeeded = $true
        }

        # Check Port 3000
        if (Test-PortActive 3000) {
            Write-Host "CARLA Server is already active on Port 3000." -ForegroundColor DarkGreen
        } else {
            Write-Host "Launching CARLA Server on Port 3000..." -ForegroundColor Green
            Start-Process -FilePath $CarlaPath -ArgumentList "-carla-rpc-port=3000"
            $launchNeeded = $true
        }

        if ($launchNeeded) { Start-Sleep -Seconds 10 }
        Invoke-Expression "python train_cone_avoidance.py --ports 2000 3000 $scenarioArg --run-name $run_name"
    }
    
    "3" {
        Write-Host "`nEvaluating Best Model..." -ForegroundColor Yellow
        python evaluate_best_model.py --name $run_name --episodes 10
    }
    "4" {
        Write-Host "`nExiting..." -ForegroundColor DarkGray
    }
    default {
        Write-Host "`nInvalid selection. Exiting..." -ForegroundColor Red
    }
}
