#!/usr/bin/env bash
# Set up LatticeIQ on a fresh Ubuntu 24.04 server (Amazon Lightsail or EC2). Run it from the cloned repository:
#
#     sudo bash deploy/aws/setup.sh
#
# Safe to run again. It adds 2 GB of swap, installs Python packages and Caddy (a web server that gets a free
# HTTPS certificate by itself), creates .env with a login password placeholder, and installs a service that
# starts the dashboard on boot. It does not start the dashboard: copy the data first (deploy/aws/README.md).
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
APP_USER="${SUDO_USER:-ubuntu}"
PORT=8600
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo: sudo bash deploy/aws/setup.sh"; exit 1; }
echo "LatticeIQ folder: $APP_DIR   runs as: $APP_USER"

# 1. Swap: the daily ranking needs about 1.8 GB for a minute; a 1 GB server swaps through it
if ! swapon --show | grep -q '^/swapfile'; then
  echo "Adding 2 GB of swap"
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# 2. System packages (Caddy comes from Ubuntu's own package list)
apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-venv python3-pip git curl caddy

# 3. Python packages, in a virtual environment owned by the app user
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/deploy/aws/requirements-server.txt"
sudo -u "$APP_USER" mkdir -p "$APP_DIR/data/clean" "$APP_DIR/data/live" "$APP_DIR/models"

# 4. Secrets file: only the app user can read it. You fill in the two values (nano .env).
if [ ! -f "$APP_DIR/.env" ]; then
  printf '%s\n' "# LatticeIQ server settings. Never commit this file." \
    "SEC_USER_AGENT=" "LATTICEIQ_PASSWORD=" > "$APP_DIR/.env"
  chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
fi
chmod 600 "$APP_DIR/.env"

# 5. The dashboard as a service: starts on boot, restarts if it stops
sed -e "s|__DIR__|$APP_DIR|g" -e "s|__USER__|$APP_USER|g" -e "s|__PORT__|$PORT|g" \
  "$APP_DIR/deploy/aws/latticeiq.service" > /etc/systemd/system/latticeiq.service
systemctl daemon-reload
systemctl enable latticeiq >/dev/null

# 6. HTTPS in front of it. Without a domain of your own, <your-ip-with-dashes>.sslip.io points at this server
#    and Caddy gets a free certificate for it. With a domain: sudo DOMAIN=app.example.com bash deploy/aws/setup.sh
IP="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')"
DOMAIN="${DOMAIN:-${IP//./-}.sslip.io}"
sed -e "s|__DOMAIN__|$DOMAIN|g" -e "s|__PORT__|$PORT|g" "$APP_DIR/deploy/aws/Caddyfile" > /etc/caddy/Caddyfile
systemctl reload caddy || systemctl restart caddy

echo
echo "Done. Next:"
echo "  1. Copy the data from your PC (deploy/aws/upload_data.ps1)."
echo "  2. nano $APP_DIR/.env   -> set SEC_USER_AGENT (name and email) and LATTICEIQ_PASSWORD"
echo "  3. sudo systemctl start latticeiq     then watch it start: journalctl -u latticeiq -f"
echo "  4. Open https://$DOMAIN  (any user name, your password)"
