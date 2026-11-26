param([string]$Config = (Join-Path $PSScriptRoot "../configs/reasoner_flow10.json"))
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$experiment = Get-Content -LiteralPath $Config -Raw -Encoding utf8 | ConvertFrom-Json
$cases = @($experiment.cases | Where-Object audio_source -eq "new_windows_offline_tts")
$paths = @($cases | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $root $_.audio)) })
foreach ($path in $paths) {
    if (!$path.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Audio output must stay inside the project." }
    if (Test-Path -LiteralPath $path) { throw "Existing recording retained: $path" }
}
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $voice = $speaker.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -eq "zh-CN" } | Select-Object -First 1
    if (!$voice) { throw "No offline zh-CN System.Speech voice is installed." }
    $speaker.SelectVoice($voice.VoiceInfo.Name)
    $entries = @()
    for ($i = 0; $i -lt $cases.Count; $i++) {
        $path = $paths[$i]
        [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path))
        $speaker.SetOutputToWaveFile($path)
        $speaker.Speak($cases[$i].reference)
        $speaker.SetOutputToNull()
        $entries += [PSCustomObject]@{id=$cases[$i].id;audio=$cases[$i].audio;reference=$cases[$i].reference;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
    }
    [PSCustomObject]@{source="Windows offline System.Speech synthetic speech, not human audio";voice=$voice.VoiceInfo.Name;created_utc=[DateTime]::UtcNow.ToString("o");count=$cases.Count;files=$entries} |
        ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 -LiteralPath (Join-Path ([IO.Path]::GetDirectoryName($paths[0])) "source.json")
    Write-Output "Created $($cases.Count) offline synthetic recordings. No online TTS was called."
} finally { $speaker.Dispose() }
