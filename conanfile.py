from conans import ConanFile, tools

class TennisanalysisConan(ConanFile):
    name = "tennisanalysis"
    version = "1.0.0"
    settings = "os", "compiler", "build_type", "arch"
    description = "<Description of Tennisanalysis here>"
    url = "None"
    license = "None"
    author = "Leon Schenk"
    topics = None

    def package(self):
        self.copy("*")

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
