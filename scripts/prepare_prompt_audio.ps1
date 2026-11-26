param(
    [Parameter(Mandatory=$true)][switch]$ApproveLocalSynthesis,
    [string]$Out = "outputs/prompt-trial-audio"
)
$ErrorActionPreference = "Stop"
if (!$ApproveLocalSynthesis) { throw "Local speech synthesis requires explicit approval." }
$cases = (Get-Content (Join-Path $PSScriptRoot "../configs/prompt_trial.json") -Raw -Encoding utf8 | ConvertFrom-Json).cases
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $voice = $speaker.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -eq "zh-CN" } | Select-Object -First 1
    if (!$voice) { throw "No local zh-CN System.Speech voice installed." }
    $speaker.SelectVoice($voice.VoiceInfo.Name)
    $files = @($cases | Where-Object wav)
    $dir = [IO.Directory]::CreateDirectory($Out).FullName
    foreach ($case in $files) {
        if (Test-Path -LiteralPath (Join-Path $dir $case.wav)) { throw "Output already exists: $($case.wav)" }
    }
    foreach ($case in $files) {
        $path = Join-Path $dir $case.wav
        $speaker.SetOutputToWaveFile($path)
        $speaker.Speak($case.reference)
        $speaker.SetOutputToNull()
    }
    [PSCustomObject]@{source="Windows offline System.Speech";voice=$voice.VoiceInfo.Name;count=$files.Count} |
        ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $dir "source.json")
} finally {
    $speaker.Dispose()
}
