@echo off

cd /d C:\

set env=csce

if defined VIRTUAL_ENV (
    echo Deactivating environment...
    call deactivate
)

if exist "%env%" (
    echo Deleting environment folder: %AMD_env%
    rmdir /s /q "%env%"
    echo -------------------------------------------------------------
    echo Virtual environment uninstalled
    echo -------------------------------------------------------------
) else (
    echo Environment "%env%" does not exist.
)

cmd /k