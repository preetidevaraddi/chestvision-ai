$taskLog = "C:\Users\Preeti Devaraddi\.gemini\antigravity-ide\brain\6d91e0f3-7f58-42c8-bbe7-0a6ee78a10ef\.system_generated\tasks\task-147.log"
while ($true) {
    if (Test-Path $taskLog) {
        $content = Get-Content $taskLog -Tail 20 -ErrorAction SilentlyContinue
        if ($content -match "Epoch 7/30") {
            Write-Host "Epoch 7 found. Waiting 30 seconds for checkpoint save..."
            Start-Sleep -Seconds 30
            Write-Host "Killing python processes..."
            Get-Process python | Stop-Process -Force
            break
        }
    }
    Start-Sleep -Seconds 60
}
