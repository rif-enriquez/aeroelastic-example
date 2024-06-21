@echo off
setlocal

:: Deleting files containing 'AeroLoad' or 'FSIDisp' in their filenames in the current directory
echo Deleting files...
for %%f in (*AeroLoad*) do (
    if exist "%%f" del "%%f"
)

for %%f in (*FSIDisp*) do (
    if exist "%%f" del "%%f"
)

for %%f in (*FS_SurfaceSection_Loads*) do (
    if exist "%%f" del "%%f"
)

echo Files deleted.
