@echo off
setlocal EnableDelayedExpansion

:: ===========================================
::  BATCH FILE PROCESSOR - github.com/RavenDS
:: ===========================================


:: Optional prefix before command (e.g. "python", "C:/Python311/python.exe", "java -jar")
set "Prefix="

:: Full path to executable/script
set "ExePath=C:/path/to/exe-or-script.py"

:: Arguments to pass
::
:: $InputPath= current file
:: $OutputPath= output
set "Command=-convert $InputPath -output $OutputPath"



:: Folder containing files to process (leave empty to be prompted at runtime)
set "Folder="

:: File extension to look for (. is added if missing)
set "Extension=.bin"

:: Output file name pattern
:: 
:: $InputName = input filename without extension
:: $Counter = 1, 2, 3..
::
:: Leave empty for default: $InputName_batch  (keeps the input file extension)
set "OutputName="



:: Set to 1 to recurse into all subfolders, 0 for top-level only
set "Recursive=0"

:: ===================
::  END OF PARAMETERS
:: ===================

:: ensure leading dot
if not "!Extension!"=="" (
    set "ExtCheck=!Extension:~0,1!"
    if not "!ExtCheck!"=="." set "Extension=.!Extension!"
)

:: prompt for folder if not set
if "!Folder!"=="" (
    set /p "Folder=Enter folder path: "
)

:: remove trailing backslash
if "!Folder:~-1!"=="\" set "Folder=!Folder:~0,-1!"
if "!Folder:~-1!"=="/" set "Folder=!Folder:~0,-1!"

:: validate folder
if not exist "!Folder!" (
    echo ERROR: Folder "!Folder!" does not exist.
    pause
    exit /b 1
)

:: set default OutputName if empty
if "!OutputName!"=="" set "OutputName=$InputName_batch"

:: build search mode
if "!Recursive!"=="1" (
    set "SearchOpt=/r "!Folder!""
) else (
    set "SearchOpt="
)

:: count files
set "FileCount=0"
if "!Recursive!"=="1" (
    for /r "!Folder!" %%F in (*!Extension!) do set /a FileCount+=1
) else (
    for %%F in ("!Folder!\*!Extension!") do set /a FileCount+=1
)

echo.
echo ========================================
echo  Batch File Processor
echo ========================================
echo  Prefix:     !Prefix!
echo  Exe:        !ExePath!
echo  Folder:     !Folder!
echo  Extension:  !Extension!
echo  Recursive:  !Recursive!
echo  Files found: !FileCount!
echo ========================================
echo.

if !FileCount! EQU 0 (
    echo No *!Extension! files found. Nothing to do.
    pause
    exit /b 0
)

:: process files
set "Current=0"
if "!Recursive!"=="1" (
    for /r "!Folder!" %%F in (*!Extension!) do (
        call :ProcessFile "%%F"
    )
) else (
    for %%F in ("!Folder!\*!Extension!") do (
        call :ProcessFile "%%F"
    )
)

echo.
echo ========================================
echo  Done. Processed !Current! / !FileCount! files.
echo ========================================
pause
exit /b 0

:: ==========
:ProcessFile
:: ==========
set /a Current+=1
set "InputPath=%~1"
set "InputName=%~n1"
set "InputExt=%~x1"
set "InputDir=%~dp1"

:: remove trailing backslash from InputDir
if "!InputDir:~-1!"=="\" set "InputDir=!InputDir:~0,-1!"

:: resolve OutputName, replace $InputName and $Counter with actual values
set "ResolvedOutputName=!OutputName:$InputName=%~n1!"
set "ResolvedOutputName=!ResolvedOutputName:$Counter=!Current!!"

:: build full output path (same folder as input, resolved name + same extension)
set "OutputPath=!InputDir!\!ResolvedOutputName!!InputExt!"

:: build final command: replace $InputPath and $OutputPath
set "FinalCmd=!Command:$InputPath=!InputPath!!"
set "FinalCmd=!FinalCmd:$OutputPath=!OutputPath!!"

echo [!Current!/!FileCount!] Processing: !InputPath!
echo             Output:     !OutputPath!
if "!Prefix!"=="" (
    echo             Command:    "!ExePath!" !FinalCmd!
) else (
    echo             Command:    !Prefix! "!ExePath!" !FinalCmd!
)
echo.

if "!Prefix!"=="" (
    "!ExePath!" !FinalCmd!
) else (
    !Prefix! "!ExePath!" !FinalCmd!
)

if !ERRORLEVEL! NEQ 0 (
    echo WARNING: Non-zero exit code (!ERRORLEVEL!) for: !InputPath!
    echo.
)
exit /b 0
