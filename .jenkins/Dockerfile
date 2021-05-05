FROM jenkins/agent:latest

RUN apt update && apt install python3 python3-pip

RUN pip3 install ninja meson conan

# tennisAnalysis requirements
RUN apt update && apt install libgtk-2.0dev
