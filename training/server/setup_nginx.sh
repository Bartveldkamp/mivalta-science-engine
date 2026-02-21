#!/bin/bash
# MiValta — Setup nginx to serve GGUF models for developer download
#
# Run on the Hetzner server:
#   bash training/server/setup_nginx.sh
#
# After setup, models are available at:
#   http://<server-ip>/models/josi-v5-interpreter-q4_k_m.gguf
#   http://<server-ip>/models/josi-v5-explainer-q4_k_m.gguf
#   http://<server-ip>/models/josi-v5-manifest.json

set -euo pipefail

MODELS_DIR="/var/www/mivalta-models"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NGINX_CONF="$SCRIPT_DIR/nginx-models.conf"

echo "============================================================"
echo "  MiValta — nginx Model Server Setup"
echo "============================================================"

# 1. Install nginx if needed
if ! command -v nginx &>/dev/null; then
    echo ""
    echo "  Installing nginx..."
    apt-get update -qq && apt-get install -y -qq nginx
fi

echo ""
echo "  nginx version: $(nginx -v 2>&1)"

# 2. Create models directory
echo ""
echo "  Creating models directory: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# 3. Install nginx config
echo "  Installing nginx config..."
cp "$NGINX_CONF" /etc/nginx/sites-available/mivalta-models
ln -sf /etc/nginx/sites-available/mivalta-models /etc/nginx/sites-enabled/mivalta-models

# Remove default site if it conflicts
if [ -f /etc/nginx/sites-enabled/default ]; then
    rm -f /etc/nginx/sites-enabled/default
    echo "  Removed default nginx site"
fi

# 4. Test and reload nginx
echo ""
echo "  Testing nginx config..."
nginx -t

echo "  Reloading nginx..."
systemctl enable nginx
systemctl reload nginx || systemctl start nginx

# 5. Show server IP
SERVER_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "============================================================"
echo "  nginx setup complete!"
echo "============================================================"
echo ""
echo "  Models directory: $MODELS_DIR"
echo "  Server URL:       http://$SERVER_IP/models/"
echo ""
echo "  Next: publish your models:"
echo "    python scripts/publish_models.py \\"
echo "      --gguf-interpreter models/gguf/josi-v5-interpreter-q4_k_m.gguf \\"
echo "      --gguf-explainer models/gguf/josi-v5-explainer-q4_k_m.gguf \\"
echo "      --upload-only"
echo ""
