#!/usr/bin/env bash
# Entrypoint for the AutoShorts container.
#
#   MODE=ui     (default) serve the Streamlit web UI on $UI_PORT
#   MODE=batch            run the pipeline once over everything in gameplay/
#
# Any explicit command passed to `docker run` wins over MODE, so
# `docker run ... autoshorts bash` still drops you into a shell.
set -euo pipefail

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

case "${MODE:-ui}" in
    ui)
        echo "Starting AutoShorts web UI on port ${UI_PORT:-8501} ..."
        exec streamlit run src/dashboard/About.py \
            --server.port="${UI_PORT:-8501}" \
            --server.address=0.0.0.0
        ;;
    batch)
        echo "Running AutoShorts pipeline over gameplay/ ..."
        exec python run.py
        ;;
    *)
        echo "Unknown MODE='${MODE}'. Expected 'ui' or 'batch'." >&2
        exit 64
        ;;
esac
