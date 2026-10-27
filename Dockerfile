# syntax=docker/dockerfile:1

FROM golang:1.27 AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/oai-to-gemini .
RUN mkdir -p /out/tmp && chmod 1777 /out/tmp

FROM scratch

COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=build --chmod=1777 /out/tmp /tmp
COPY --from=build /out/oai-to-gemini /oai-to-gemini

ENV TMPDIR=/tmp
WORKDIR /app
EXPOSE 8080
USER 65532:65532
ENTRYPOINT ["/oai-to-gemini"]
