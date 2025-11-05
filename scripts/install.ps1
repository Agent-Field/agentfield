# AgentField CLI Installer for Windows
# Usage: iwr -useb https://agentfield.ai/install.ps1 | iex
# Version pinning: $env:VERSION="v1.0.0"; iwr -useb https://agentfield.ai/install.ps1 | iex

$ErrorActionPreference = "Stop"

# Configuration
$Repo = "Agent-Field/agentfield"
$InstallDir = if ($env:AGENTFIELD_INSTALL_DIR) { $env:AGENTFIELD_INSTALL_DIR } else { "$env:USERPROFILE\.agentfield\bin" }
$Version = if ($env:VERSION) { $env:VERSION } else { "latest" }
$Verbose = if ($env:VERBOSE -eq "1") { $true } else { $false }
$SkipPathConfig = if ($env:SKIP_PATH_CONFIG -eq "1") { $true } else { $false }

# Colors
function Write-ColorOutput {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message,
        [string]$Color = "White",
        [string]$Prefix = ""
    )

    if ($Prefix) {
        Write-Host -NoNewline "[$Prefix] " -ForegroundColor $Color
        Write-Host $Message
    } else {
        Write-Host $Message -ForegroundColor $Color
    }
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput -Message $Message -Color "Cyan" -Prefix "INFO"
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput -Message $Message -Color "Green" -Prefix "SUCCESS"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput -Message $Message -Color "Red" -Prefix "ERROR"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput -Message $Message -Color "Yellow" -Prefix "WARNING"
}

function Write-Verbose {
    param([string]$Message)
    if ($Verbose) {
        Write-ColorOutput -Message $Message -Color "DarkCyan" -Prefix "VERBOSE"
    }
}

function Write-Banner {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║           AgentField CLI Installer (Windows)             ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

# Detect architecture
function Get-Architecture {
    $arch = $env:PROCESSOR_ARCHITECTURE

    switch ($arch) {
        "AMD64" { return "amd64" }
        "ARM64" { return "arm64" }
        default {
            Write-Error "Unsupported architecture: $arch"
            Write-Info "Supported architectures:"
            Write-Info "  - AMD64 (x86_64)"
            Write-Info "  - ARM64"
            Write-Info ""
            Write-Info "Please open an issue: https://github.com/$Repo/issues"
            exit 1
        }
    }
}

# Get latest version from GitHub API
function Get-LatestVersion {
    Write-Verbose "Fetching latest version from GitHub API..."

    $latestUrl = "https://api.github.com/repos/$Repo/releases/latest"

    try {
        $response = Invoke-RestMethod -Uri $latestUrl -Method Get
        $version = $response.tag_name

        if ([string]::IsNullOrEmpty($version)) {
            throw "Failed to parse version from API response"
        }

        return $version
    }
    catch {
        Write-Error "Failed to determine latest version from GitHub API"
        Write-Info "You can manually specify a version: `$env:VERSION=`"v1.0.0`"; .\install.ps1"
        Write-Error $_.Exception.Message
        exit 1
    }
}

# Download file
function Download-File {
    param(
        [string]$Url,
        [string]$OutputPath
    )

    Write-Verbose "Downloading: $Url"
    Write-Verbose "To: $OutputPath"

    try {
        $ProgressPreference = if ($Verbose) { "Continue" } else { "SilentlyContinue" }
        Invoke-WebRequest -Uri $Url -OutFile $OutputPath -UseBasicParsing
    }
    catch {
        Write-Error "Download failed: $Url"
        Write-Error $_.Exception.Message
        exit 1
    }

    if (-not (Test-Path $OutputPath)) {
        Write-Error "Download failed: file not created at $OutputPath"
        exit 1
    }
}

# Verify checksum
function Test-Checksum {
    param(
        [string]$BinaryPath,
        [string]$ChecksumsFile,
        [string]$BinaryName
    )

    Write-Info "Verifying checksum..."
    Write-Verbose "Binary: $BinaryPath"
    Write-Verbose "Checksums file: $ChecksumsFile"
    Write-Verbose "Binary name: $BinaryName"

    # Read checksums file
    $checksumContent = Get-Content $ChecksumsFile -Raw
    $checksumLines = $checksumContent -split "`n"

    # Find the line with our binary
    $checksumLine = $checksumLines | Where-Object { $_ -match [regex]::Escape($BinaryName) }

    if ([string]::IsNullOrEmpty($checksumLine)) {
        Write-Error "Could not find checksum for $BinaryName in checksums file"
        Write-Verbose "Checksums file content:"
        if ($Verbose) {
            Write-Host $checksumContent
        }
        exit 1
    }

    # Extract expected checksum (format: "hash  filename")
    $expectedChecksum = ($checksumLine -split '\s+')[0]

    Write-Verbose "Expected checksum: $expectedChecksum"

    # Calculate actual checksum
    $actualChecksum = (Get-FileHash -Path $BinaryPath -Algorithm SHA256).Hash.ToLower()

    Write-Verbose "Actual checksum: $actualChecksum"

    if ($actualChecksum -ne $expectedChecksum) {
        Write-Error "Checksum verification failed!"
        Write-Error "Expected: $expectedChecksum"
        Write-Error "Got:      $actualChecksum"
        Write-Error ""
        Write-Error "This may indicate a corrupted download or security issue."
        Write-Info "Please try again or report this issue:"
        Write-Info "  https://github.com/$Repo/issues"
        exit 1
    }

    Write-Success "Checksum verified"
}

# Install binary
function Install-Binary {
    param(
        [string]$BinaryPath,
        [string]$InstallDir
    )

    Write-Info "Installing to $InstallDir"

    # Create install directory
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    # Copy binary
    $targetPath = Join-Path $InstallDir "agentfield.exe"
    Copy-Item -Path $BinaryPath -Destination $targetPath -Force

    Write-Success "Binary installed to $targetPath"
}

# Configure PATH
function Set-PathConfiguration {
    param(
        [string]$InstallDir
    )

    if ($SkipPathConfig) {
        Write-Info "Skipping PATH configuration (SKIP_PATH_CONFIG=1)"
        return
    }

    Write-Info "Configuring PATH..."

    # Get current user PATH
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")

    # Check if already in PATH
    if ($userPath -like "*$InstallDir*") {
        Write-Info "PATH already configured"
        return
    }

    # Add to PATH
    $newPath = if ($userPath) { "$userPath;$InstallDir" } else { $InstallDir }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")

    # Update PATH for current session
    $env:Path = "$env:Path;$InstallDir"

    Write-Success "PATH configured"
    Write-Info ""
    Write-Info "PATH has been updated for future sessions."
    Write-Info "For this session, restart your terminal or run:"
    Write-Host "  `$env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')" -ForegroundColor Cyan
}

# Verify installation
function Test-Installation {
    param(
        [string]$InstallDir
    )

    Write-Info "Verifying installation..."

    $binaryPath = Join-Path $InstallDir "agentfield.exe"

    if (Test-Path $binaryPath) {
        Write-Success "Installation verified"

        # Try to get version
        try {
            $versionOutput = & $binaryPath --version 2>&1
            Write-Verbose "Version output: $versionOutput"
        }
        catch {
            # Ignore version check errors
        }
    }
    else {
        Write-Error "Installation verification failed"
        Write-Error "Binary not found: $binaryPath"
        exit 1
    }
}

# Print success message
function Write-SuccessMessage {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║  AgentField CLI installed successfully!                      ║" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host ""
    Write-Host "  1. Restart your terminal or refresh PATH:" -ForegroundColor White
    Write-Host "     `$env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  2. Verify installation:" -ForegroundColor White
    Write-Host "     agentfield --version" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  3. Initialize your first agent:" -ForegroundColor White
    Write-Host "     agentfield init my-agent" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Resources:" -ForegroundColor White
    Write-Host "  Documentation: https://agentfield.ai/docs" -ForegroundColor Blue
    Write-Host "  GitHub:        https://github.com/$Repo" -ForegroundColor Blue
    Write-Host "  Support:       https://github.com/$Repo/issues" -ForegroundColor Blue
    Write-Host ""
}

# Main installation flow
function Main {
    Write-Banner

    # Detect platform
    $arch = Get-Architecture
    $os = "windows"

    Write-Info "Detected platform: $os-$arch"

    # Determine version
    if ($Version -eq "latest") {
        $script:Version = Get-LatestVersion
    }

    Write-Info "Installing version: $Version"

    # Construct binary name and URL
    $binaryName = "agentfield-$os-$arch.exe"
    $downloadUrl = "https://github.com/$Repo/releases/download/$Version/$binaryName"
    $checksumsUrl = "https://github.com/$Repo/releases/download/$Version/checksums.txt"

    Write-Verbose "Binary name: $binaryName"
    Write-Verbose "Download URL: $downloadUrl"
    Write-Verbose "Checksums URL: $checksumsUrl"

    # Create temporary directory
    $tempDir = Join-Path $env:TEMP "agentfield-install-$(Get-Random)"
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

    try {
        # Download binary
        Write-Info "Downloading binary..."
        $binaryPath = Join-Path $tempDir $binaryName
        Download-File -Url $downloadUrl -OutputPath $binaryPath
        Write-Success "Binary downloaded"

        # Download checksums
        Write-Info "Downloading checksums..."
        $checksumsPath = Join-Path $tempDir "checksums.txt"
        Download-File -Url $checksumsUrl -OutputPath $checksumsPath
        Write-Success "Checksums downloaded"

        # Verify checksum
        Test-Checksum -BinaryPath $binaryPath -ChecksumsFile $checksumsPath -BinaryName $binaryName

        # Install binary
        Install-Binary -BinaryPath $binaryPath -InstallDir $InstallDir

        # Configure PATH
        Set-PathConfiguration -InstallDir $InstallDir

        # Verify installation
        Test-Installation -InstallDir $InstallDir

        # Print success message
        Write-SuccessMessage
    }
    finally {
        # Cleanup temporary directory
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# Run main function
Main
