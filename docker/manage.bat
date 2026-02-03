@echo off
REM Endee RAG System - Docker Management Script for Windows

set COMPOSE_FILE=docker/docker-compose.yml

if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="status" goto status
if "%1"=="logs" goto logs
if "%1"=="health" goto health
goto usage

:start
echo Starting Endee service...
docker-compose -f %COMPOSE_FILE% up -d
echo Endee service started. Access at http://localhost:8080
goto end

:stop
echo Stopping Endee service...
docker-compose -f %COMPOSE_FILE% down
echo Endee service stopped.
goto end

:restart
echo Restarting Endee service...
docker-compose -f %COMPOSE_FILE% down
docker-compose -f %COMPOSE_FILE% up -d
echo Endee service restarted.
goto end

:status
echo Checking Endee service status...
docker-compose -f %COMPOSE_FILE% ps
goto end

:logs
echo Showing Endee service logs...
docker-compose -f %COMPOSE_FILE% logs -f endee
goto end

:health
echo Checking Endee health...
curl -f http://localhost:8080/health
if errorlevel 1 echo Health check failed
goto end

:usage
echo Usage: %0 {start^|stop^|restart^|status^|logs^|health}
echo.
echo Commands:
echo   start   - Start Endee service
echo   stop    - Stop Endee service
echo   restart - Restart Endee service
echo   status  - Show service status
echo   logs    - Show service logs
echo   health  - Check service health

:end