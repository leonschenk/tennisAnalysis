# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.166.1/containers/cpp/.devcontainer/base.Dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu

# [Optional] Uncomment this section to install additional packages.
RUN apt update && apt -y install --no-install-recommends python3 python3-setuptools python3-pip

# install gcc
RUN apt update && apt -y install --no-install-recommends mingw-w64

# install libgtk dependency
#RUN apt-get update && apt-get -y install --no-install-recommends libgtk2.0-dev

RUN pip3 install ninja meson conan
