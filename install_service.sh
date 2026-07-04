#!/usr/bin/env bash
# install_service.sh — Install and enable Evolution Engine systemd service
set -euo pipefail

SERVICE_NAME="evolution-engine"
SERVICE_DIR="$HOME/.config/systemd/user"
PROJECT_DIR="/mnt/c/Users/reese/Projects/Evolution Engine"

echo "══════════════════════════════════════════════════════════════════"
echo " Evolution Engine — Service Installation"
echo "══════════════════════════════════════════════════════════════════"

# Create systemd user directory if it doesn't exist
mkdir -p "$SERVICE_DIR"

# Copy service file
echo ""
echo "Installing service file to $SERVICE_DIR/"
cp "$PROJECT_DIR/evolution-engine.service" "$SERVICE_DIR/${SERVICE_NAME}.service"

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl --user daemon-reload

# Enable the service
echo "Enabling $SERVICE_NAME service..."
systemctl --user enable "$SERVICE_NAME"

# Start the service
echo "Starting $SERVICE_NAME service..."
systemctl --user start "$SERVICE_NAME"

# Wait a moment for startup
sleep 2

# Show status
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo " Service Status"
echo "══════════════════════════════════════════════════════════════════"
systemctl --user status "$SERVICE_NAME" || true

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo " Useful Commands"
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo "  View logs:        journalctl --user -u $SERVICE_NAME -f"
echo "  Stop service:     systemctl --user stop $SERVICE_NAME"
echo "  Start service:    systemctl --user start $SERVICE_NAME"
echo "  Restart service:  systemctl --user restart $SERVICE_NAME"
echo "  Disable service:  systemctl --user disable $SERVICE_NAME"
echo "  Check status:     ./status.sh"
echo ""
