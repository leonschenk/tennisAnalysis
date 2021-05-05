from conans import ConanFile, tools, Meson
import os

class TennisAnalysisFile(ConanFile):
    generators = "pkg_config"
    requires = "opencv/4.5.2"
    settings = "os", "compiler", "arch", "build_type"

    def build(self):
        meson = Meson(self)
        meson.configure(source_folder="src", build_folder="build")
        meson.build()

        
