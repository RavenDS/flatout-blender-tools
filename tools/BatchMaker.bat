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
:: $InputPath  = current file being processed
:: $OutputPath = output file path (default: $InputName_batch)
::
set "Command=-convert $InputPath -output $OutputPath"



:: Folder (leave empty to be prompted at runtime)
set "Folder="

:: File extension to look for (. is added if missing)
set "Extension=.bin"

:: Set to 1 to recurse into all subfolders, 0 for top-level only
set "Recursive=0"

:: Output file name (optional, used for $OutputPath)
:: $InputName = input filename without extension
:: Leave empty for default: $InputName_batch
set "OutputName="


:: ===================
::  END OF PARAMETERS
:: ===================

set "ExtCheck=!Extension:~0,1!"
if not "!ExtCheck!"=="." set "Extension=.!Extension!"

if "!Folder!"=="" set /p "Folder=Enter folder path: "
if "!Folder:~-1!"=="\" set "Folder=!Folder:~0,-1!"
if "!Folder:~-1!"=="/" set "Folder=!Folder:~0,-1!"
if not exist "!Folder!" echo ERROR: Folder "!Folder!" not found. & pause & exit /b 1

if "!OutputName!"=="" set "OutputName=$InputName_batch"

set "FileCount=0"
if "!Recursive!"=="1" ( for /r "!Folder!" %%F in (*!Extension!) do set /a FileCount+=1
) else ( for %%F in ("!Folder!\*!Extension!") do set /a FileCount+=1 )

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

if !FileCount! EQU 0 echo No *!Extension! files found. & pause & exit /b 0

set "Current=0"
if "!Recursive!"=="1" ( for /r "!Folder!" %%F in (*!Extension!) do call :Run "%%F"
) else ( for %%F in ("!Folder!\*!Extension!") do call :Run "%%F" )

echo.
echo  Done. !Current! / !FileCount! files.
echo ========================================
pause
exit /b 0


:Run
set /a Current+=1
set "InputPath=%~1"
set "InputName=%~n1"
set "InputExt=%~x1"
set "InputDir=%~dp1"
if "!InputDir:~-1!"=="\" set "InputDir=!InputDir:~0,-1!"

set "ResolvedOutputName=!OutputName!"
set "ResolvedOutputName=!ResolvedOutputName:$InputName=%InputName%!"
set "ResolvedOutputName=!ResolvedOutputName:$Counter=%Current%!"
set "OutputPath=!InputDir!\!ResolvedOutputName!!InputExt!"

set "FinalCmd=!Command!"
set "FinalCmd=!FinalCmd:$InputPath=%InputPath%!"
set "FinalCmd=!FinalCmd:$OutputPath=%OutputPath%!"

set "RunCmd="!ExePath!" !FinalCmd!"
if not "!Prefix!"=="" set "RunCmd=!Prefix! !RunCmd!"

echo [!Current!/!FileCount!] !InputName!!InputExt!
!RunCmd!
exit /b 0
