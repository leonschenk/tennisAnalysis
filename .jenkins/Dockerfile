FROM jenkins/agent:latest

USER root
RUN apt update && apt install -y python3 python3-pip

# tennisAnalysis requirements
RUN apt update && apt install -y libgtk2.0-dev

RUN pip3 install ninja meson conan

USER jenkins
