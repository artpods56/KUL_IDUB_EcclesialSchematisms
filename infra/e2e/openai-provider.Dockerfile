FROM python:3.14.7-slim-bookworm@sha256:9ab8d9c8514b44f90cf0029dd42fdd7e9e211e639c8b995304cc04568dee900f

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/grafy-e2e-provider

RUN install -d -m 0755 -o 65532 -g 65532 ./tls

COPY --chown=65532:65532 --chmod=0555 infra/e2e/openai-provider/server.py ./server.py
COPY --chown=65532:65532 --chmod=0444 infra/e2e/tls/server.crt ./tls/server.crt
COPY --chown=65532:65532 --chmod=0400 infra/e2e/tls/server.key ./tls/server.key

USER 65532:65532

EXPOSE 8443

HEALTHCHECK --interval=2s --timeout=2s --retries=15 \
    CMD ["python", "-c", "import ssl,urllib.request; urllib.request.urlopen('https://127.0.0.1:8443/healthz', context=ssl.create_default_context(cafile='/opt/grafy-e2e-provider/tls/server.crt'), timeout=1)"]

ENTRYPOINT ["python", "-I", "/opt/grafy-e2e-provider/server.py"]
CMD ["--port", "8443", "--certificate", "/opt/grafy-e2e-provider/tls/server.crt", "--private-key", "/opt/grafy-e2e-provider/tls/server.key", "--api-key", "grafy-e2e-provider-key"]
