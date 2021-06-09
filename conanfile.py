from conans import ConanFile
from conan.tools.meson import Meson

import os

class TennisAnalysisFile(ConanFile):
    name = "TennisAnalysis"
    version = "1.0.0"
    generators = ["PkgConfig", "MesonToolchain"]
    requires = "opencv/4.5.2"
    build_requires = "meson/0.58.1"
    settings = "os", "compiler", "arch", "build_type"
    exports_sources = "src/*"

    def configure(self):
        self.options["opencv"].with_openexr = False
        self.options["opencv"].dnn = False

    def layout(self):
        self.folders.source = ""
        self.folders.build = "build"
        self.folders.generators = os.path.join(self.folders.build, "generators")
        self.folders.package = "package"
        
        self.cpp.build.bindirs = ["."]

    def build(self):
        meson = Meson(self, build_folder=None)
        meson.configure(source_folder="src")
        meson.build()

    def package(self):
        meson = Meson(self, build_folder=None)
        meson.set_prefix()
        meson.install()
        
