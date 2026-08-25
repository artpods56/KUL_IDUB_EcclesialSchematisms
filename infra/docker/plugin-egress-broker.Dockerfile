FROM python:3.14.0-slim-trixie@sha256:0aecac02dc3d4c5dbb024b753af084cafe41f5416e02193f1ce345d671ec966e

COPY apps/api/src/grafy_api/plugin_egress_broker.py \
    /opt/grafy/bin/grafy-plugin-egress-broker

RUN chmod 0555 /opt/grafy/bin/grafy-plugin-egress-broker

USER 65532:65532
