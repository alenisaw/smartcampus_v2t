FROM ubuntu:latest
LABEL authors="alenk"

ENTRYPOINT ["top", "-b"]