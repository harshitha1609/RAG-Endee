#!/bin/bash

# Endee RAG System - Docker Management Script

COMPOSE_FILE="docker/docker-compose.yml"

case "$1" in
    start)
        echo "Starting Endee service..."
        docker-compose -f $COMPOSE_FILE up -d
        echo "Endee service started. Access at http://localhost:8080"
        ;;
    stop)
        echo "Stopping Endee service..."
        docker-compose -f $COMPOSE_FILE down
        echo "Endee service stopped."
        ;;
    restart)
        echo "Restarting Endee service..."
        docker-compose -f $COMPOSE_FILE down
        docker-compose -f $COMPOSE_FILE up -d
        echo "Endee service restarted."
        ;;
    status)
        echo "Checking Endee service status..."
        docker-compose -f $COMPOSE_FILE ps
        ;;
    logs)
        echo "Showing Endee service logs..."
        docker-compose -f $COMPOSE_FILE logs -f endee
        ;;
    health)
        echo "Checking Endee health..."
        curl -f http://localhost:8080/health || echo "Health check failed"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|health}"
        echo ""
        echo "Commands:"
        echo "  start   - Start Endee service"
        echo "  stop    - Stop Endee service"
        echo "  restart - Restart Endee service"
        echo "  status  - Show service status"
        echo "  logs    - Show service logs"
        echo "  health  - Check service health"
        exit 1
        ;;
esac