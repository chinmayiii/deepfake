@echo off
setlocal

set "SKIP_MODEL_LOAD=1"
set "PYTHON_EXE=C:\Users\smile\AppData\Local\Programs\Python\Python313\python.exe"
if exist "%PYTHON_EXE%" (
	"%PYTHON_EXE%" -W ignore::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v
) else (
	python -W ignore::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v
)

endlocal