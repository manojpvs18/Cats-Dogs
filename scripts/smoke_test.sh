#!/usr/bin/env bash
# Post-deployment smoke test: health check + one prediction call.
# Exits non-zero (failing the CD pipeline) if either check fails.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "== Smoke test: health check =="
health_status=$(curl -sf -o /tmp/health.json -w "%{http_code}" "${BASE_URL}/health")
if [ "$health_status" != "200" ]; then
  echo "FAIL: /health returned $health_status"
  exit 1
fi
cat /tmp/health.json
echo

echo "== Smoke test: prediction call =="
python3 - <<'EOF'
from PIL import Image
img = Image.new("RGB", (224, 224), color=(100, 150, 200))
img.save("/tmp/smoke_test.jpg")
EOF

predict_status=$(curl -sf -o /tmp/predict.json -w "%{http_code}" \
  -X POST -F "file=@/tmp/smoke_test.jpg" "${BASE_URL}/predict")
if [ "$predict_status" != "200" ]; then
  echo "FAIL: /predict returned $predict_status"
  exit 1
fi
cat /tmp/predict.json
echo

echo "Smoke tests passed."
