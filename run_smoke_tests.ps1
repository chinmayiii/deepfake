$env:SKIP_MODEL_LOAD = "1"

$pythonExe = "C:\Users\smile\AppData\Local\Programs\Python\Python313\python.exe"
if (Test-Path $pythonExe) {
    & $pythonExe -W ignore::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v
} else {
    python -W ignore::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v
}