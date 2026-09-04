FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG RVC_REF=81eed5e8f68b6bed1789f682fe78cdd324495afc

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg git libsndfile1 python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/rvc-venv && python3 -m venv /opt/api-venv
ENV PATH="/opt/api-venv/bin:${PATH}"

RUN git init /opt/rvc \
    && git -C /opt/rvc remote add origin \
       https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git \
    && git -C /opt/rvc fetch --filter=blob:none --depth 1 origin "${RVC_REF}" \
    && git -C /opt/rvc checkout --detach FETCH_HEAD \
    && sed -i 's#https://mirrors.pku.edu.cn/pypi/simple#https://pypi.org/simple#g' \
       /opt/rvc/requirments_cu128_py312.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rvc-venv/bin/pip install \
    torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rvc-venv/bin/pip install \
    -r /opt/rvc/requirments_cu128_py312.txt huggingface_hub

WORKDIR /app
COPY engine/requirements.txt /app/engine/requirements.txt
COPY engine/requirements-audio.txt /app/engine/requirements-audio.txt
RUN pip install --no-cache-dir -r /app/engine/requirements.txt \
    && /opt/rvc-venv/bin/pip install --no-cache-dir -r /app/engine/requirements-audio.txt \
    && cd /opt/rvc \
    && /opt/rvc-venv/bin/hf download lj1995/VoiceConversionWebUI --revision main \
       --include "hubert_base/*" --local-dir assets \
    && /opt/rvc-venv/bin/hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
       --local-dir assets/rmvpe

ENV TORCH_HOME=/opt/audio-models/torch
RUN mkdir -p "${TORCH_HOME}" \
    && /opt/rvc-venv/bin/python -c "from demucs.pretrained import get_model; get_model('htdemucs')" \
    && /opt/rvc-venv/bin/python -c "from silero_vad import load_silero_vad; load_silero_vad(onnx=True)"

COPY engine /app/engine

RUN useradd --create-home --uid 10001 voicemerge \
    && mkdir -p /var/lib/voicemerge/models /var/lib/voicemerge/jobs \
    && chown -R voicemerge:voicemerge /var/lib/voicemerge /app /opt/rvc /opt/audio-models

USER voicemerge
ENV PYTHONUNBUFFERED=1 \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
    VOICEMERGE_RVC_ROOT=/opt/rvc \
    VOICEMERGE_RVC_PYTHON=/opt/rvc-venv/bin/python \
    VOICEMERGE_MODEL_CACHE=/var/lib/voicemerge/models \
    VOICEMERGE_JOB_ROOT=/var/lib/voicemerge/jobs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl --fail -H "Authorization: Bearer ${VOICEMERGE_API_TOKEN}" http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["uvicorn", "engine.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
