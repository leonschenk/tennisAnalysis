# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.166.1/containers/cpp/.devcontainer/base.Dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu

# [Optional] Uncomment this section to install additional packages.
RUN apt update && apt -y install --no-install-recommends python3 python3-setuptools python3-pip

# install gcc
RUN apt update && apt -y install --no-install-recommends mingw-w64

COPY conan/profiles/* /home/vscode/.conan/profiles/

# install conan
RUN git clone https://github.com/conan-io/conan.git conan_src && cd conan_src && python3 -m pip install -e .

