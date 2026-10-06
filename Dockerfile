# syntax=docker/dockerfile:1.6
#
# chora-qgen Dockerfile — the qgen ADK agent crew (qgen_question generator,
# qgen_critic, qgen_renderer). ONE image, THREE binaries; the container
# command selects which runs (e.g. chora-qgen-critic runs /app/qgen_critic).
#
# Build context = this repository. The crew is standalone; shared Chora
# modules (chora-adk-common, chora-common) are resolved through the Go module
# proxy, not a workspace.
#
# Standard invocation:
#   docker buildx build --platform=linux/amd64 \
#     -f Dockerfile \
#     --build-arg SERVICE_NAME=chora-qgen \
#     --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
#     --build-arg BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
#     -t walfa/chora-qgen:latest \
#     .

ARG GO_VERSION=1.26.6
ARG ALPINE_VERSION=3.23
ARG SERVICE_NAME=chora-qgen
ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown

############################
# Stage 1 — build + test
############################
FROM golang:${GO_VERSION}-alpine${ALPINE_VERSION} AS builder

ARG SERVICE_NAME
ARG GIT_SHA
ARG BUILD_TIME
ARG TARGETARCH

WORKDIR /src

RUN apk add --no-cache ca-certificates git

COPY . .

RUN go mod download

ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=${TARGETARCH}

# Build every binary and run the suite. An image that ships with a failing
# test is worse than no image: the failure would surface only at deploy time.
RUN go build -trimpath -ldflags "-s -w" -o /out/qgen_question ./cmd/qgen_question \
    && go build -trimpath -ldflags "-s -w" -o /out/qgen_critic ./cmd/qgen_critic \
    && go build -trimpath -ldflags "-s -w" -o /out/qgen_renderer ./cmd/qgen_renderer \
    && go vet ./... \
    && go test ./...

############################
# Stage 2 — runtime
############################
FROM alpine:${ALPINE_VERSION}

ARG SERVICE_NAME
ARG GIT_SHA
ARG BUILD_TIME

RUN apk add --no-cache ca-certificates \
    && addgroup -S app \
    && adduser -S -G app app

LABEL org.opencontainers.image.title="${SERVICE_NAME}" \
      org.opencontainers.image.source="https://github.com/apollo-chora/chora-qgen" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_TIME}" \
      org.opencontainers.image.vendor="Chora Platform" \
      org.opencontainers.image.licenses="UNLICENSED" \
      io.chora.service="${SERVICE_NAME}" \
      io.chora.git-sha="${GIT_SHA}" \
      io.chora.build-time="${BUILD_TIME}"

WORKDIR /app

COPY --from=builder /out/qgen_question /app/qgen_question
COPY --from=builder /out/qgen_critic /app/qgen_critic
COPY --from=builder /out/qgen_renderer /app/qgen_renderer

# Default ENTRYPOINT is the generator; the per-member container `command`
# overrides it (e.g. chora-qgen-critic runs /app/qgen_critic).
USER app:app
ENTRYPOINT ["/app/qgen_question"]
