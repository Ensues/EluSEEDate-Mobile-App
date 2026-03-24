param(
    [switch]$WriteLocalProperties
)

$ErrorActionPreference = 'Stop'

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "== $Text =="
}

function Write-Info {
    param([string]$Text)
    Write-Host "[INFO] $Text"
}

function Write-Warn {
    param([string]$Text)
    Write-Host "[WARN] $Text" -ForegroundColor Yellow
}

function Write-Ok {
    param([string]$Text)
    Write-Host "[OK]   $Text" -ForegroundColor Green
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$androidDir = Join-Path $repoRoot "android"
$localPropertiesPath = Join-Path $androidDir "local.properties"

Write-Section "Java Check"

$javaCmd = Get-Command java -ErrorAction SilentlyContinue
if (-not $javaCmd) {
    Write-Warn "'java' command not found in PATH. Install JDK 17 and set JAVA_HOME + PATH."
} else {
    # java -version writes to stderr; use cmd redirection to avoid PowerShell treating it as an error record.
    $javaVersionOutput = cmd /c "java -version 2>&1"
    $javaVersionText = ($javaVersionOutput | Out-String)
    Write-Info $javaVersionText.Trim()

    if ($javaVersionText -match 'version\s+"(?<v>\d+)(\.|\")') {
        $major = [int]$Matches['v']
        if ($major -eq 17) {
            Write-Ok "Java major version is 17 (recommended)."
        } else {
            Write-Warn "Java major version is $major. JDK 17 is recommended for this project."
        }
    } else {
        Write-Warn "Could not parse Java version output."
    }
}

$javaHome = $env:JAVA_HOME
$javaHomeSource = "process"
if (-not $javaHome) {
    $javaHomeUser = [Environment]::GetEnvironmentVariable('JAVA_HOME', 'User')
    if ($javaHomeUser) {
        $javaHome = $javaHomeUser
        $javaHomeSource = "user"
    }
}
if (-not $javaHome) {
    $javaHomeMachine = [Environment]::GetEnvironmentVariable('JAVA_HOME', 'Machine')
    if ($javaHomeMachine) {
        $javaHome = $javaHomeMachine
        $javaHomeSource = "machine"
    }
}

if ($javaHome) {
    Write-Info "JAVA_HOME=$javaHome"
    if (Test-Path $javaHome) {
        Write-Ok "JAVA_HOME directory exists."
        if ($javaHomeSource -ne "process") {
            Write-Info "JAVA_HOME is set in $javaHomeSource environment variables. Restart terminal/VS Code to load it into current shell."
        }
    } else {
        Write-Warn "JAVA_HOME points to a non-existing directory."
    }
} else {
    Write-Warn "JAVA_HOME is not set."
}

Write-Section "Android SDK Path Detection"

$sdkCandidates = @()
if ($env:ANDROID_HOME) { $sdkCandidates += $env:ANDROID_HOME }
if ($env:ANDROID_SDK_ROOT) { $sdkCandidates += $env:ANDROID_SDK_ROOT }
$sdkCandidates += (Join-Path $env:LOCALAPPDATA "Android\Sdk")

$sdkPath = $null
foreach ($candidate in $sdkCandidates | Select-Object -Unique) {
    if ($candidate -and (Test-Path $candidate)) {
        $sdkPath = $candidate
        break
    }
}

if (-not $sdkPath) {
    Write-Warn "Android SDK path not found. Set ANDROID_HOME or install SDK in default location."
    exit 1
}

Write-Ok "Using Android SDK: $sdkPath"

Write-Section "Required Components Check"

$requiredPaths = @(
    @{ Label = "platforms/android-35"; Path = (Join-Path $sdkPath "platforms\android-35") },
    @{ Label = "build-tools/35.0.0"; Path = (Join-Path $sdkPath "build-tools\35.0.0") },
    @{ Label = "platform-tools"; Path = (Join-Path $sdkPath "platform-tools") },
    @{ Label = "command-line tools"; Path = (Join-Path $sdkPath "cmdline-tools") }
)

$missing = @()
foreach ($item in $requiredPaths) {
    if (Test-Path $item.Path) {
        Write-Ok "Found $($item.Label)"
    } else {
        Write-Warn "Missing $($item.Label)"
        $missing += $item.Label
    }
}

$adbExe = Join-Path $sdkPath "platform-tools\adb.exe"
if (Test-Path $adbExe) {
    Write-Ok "ADB found at $adbExe"
} else {
    Write-Warn "ADB executable not found at expected path."
}

Write-Section "local.properties"

if ($WriteLocalProperties) {
    $escapedSdk = $sdkPath.Replace('\', '\\')
    $content = "sdk.dir=$escapedSdk"
    Set-Content -Path $localPropertiesPath -Value $content -Encoding ASCII
    Write-Ok "Wrote $localPropertiesPath"
} else {
    if (Test-Path $localPropertiesPath) {
        Write-Ok "Found existing android/local.properties"
        Write-Info ((Get-Content $localPropertiesPath -ErrorAction SilentlyContinue) -join "`n")
    } else {
        Write-Warn "android/local.properties not found. Run this script with -WriteLocalProperties to create it."
    }
}

Write-Section "Summary"
if ($missing.Count -eq 0) {
    Write-Ok "All required SDK components are present."
} else {
    Write-Warn "Missing components: $($missing -join ', ')"
    if ($missing -contains "build-tools/35.0.0") {
        $buildToolsRoot = Join-Path $sdkPath "build-tools"
        if (Test-Path $buildToolsRoot) {
            $installedBuildTools = Get-ChildItem -Path $buildToolsRoot -Directory | Select-Object -ExpandProperty Name
            if ($installedBuildTools) {
                Write-Info "Installed build-tools versions: $($installedBuildTools -join ', ')"
            }
        }
    }
    Write-Warn "Install missing items via Android Studio > Tools > SDK Manager."
}

Write-Host ""
Write-Host "Done."
