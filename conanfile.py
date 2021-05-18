from conans import ConanFile
from mesonbuild.build import Meson

class TennisAnalysisFile(ConanFile):
    generators = ["pkg_config", "MesonToolchain"]
    requires = "opencv/4.5.2"
    settings = "os", "compiler", "arch", "build_type"

    def configure(self):
        self.folders.generators = "generators"

        self.options["opencv"].with_openexr = False
        self.options["opencv"].dnn = False

    def build(self):
        meson = Meson(self)
        meson.configure(source_folder="src")
        meson.build()

        
