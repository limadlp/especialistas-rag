#!/bin/zsh
set -e

cd "${0:A:h}"
APP_URL="http://127.0.0.1:8001"
LM_CLI="$HOME/.lmstudio/bin/lms"

if curl --silent --fail --max-time 2 "$APP_URL/api/specialists" >/dev/null; then
  open "$APP_URL"
  exit 0
fi

if ! curl --silent --fail --max-time 2 http://127.0.0.1:1234/v1/models >/dev/null; then
  if [[ -x "$LM_CLI" ]]; then
    "$LM_CLI" server start --bind 127.0.0.1 --port 1234 || echo "Inicie o servidor local no LM Studio para conversar."
  else
    echo "LM Studio não encontrado. Inicie seu servidor local na porta 1234."
  fi
fi

(
  for attempt in {1..80}; do
    if curl --silent --fail --max-time 1 "$APP_URL/api/specialists" >/dev/null; then
      open "$APP_URL"
      exit 0
    fi
    sleep 0.25
  done
) &

exec .venv/bin/especialistas-rag platform-serve
