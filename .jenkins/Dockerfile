FROM jenkins/agent:latest

USER root
RUN apt update && apt install -y python3 python3-pip

# tennisAnalysis requirements
RUN apt update && apt install -y libgtk-2.0dev

USER jenkins
RUN pip3 install ninja meson conan
