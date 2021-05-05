FROM jenkins/agent:latest

RUN sudo apt update && sudo apt install python3 python3-pip

RUN pip3 install ninja meson conan

# tennisAnalysis requirements
RUN sudo apt update && sudo apt install libgtk-2.0dev
