"""Bundle configuration templates without duplicating their source files."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def get_output_mapping(self):
        mapping = super().get_output_mapping()
        for source in sorted(Path("configs").glob("*.yaml")):
            mapping[str(Path(self.build_lib) / "pvfit" / "_defaults" / source.name)] = (
                str(source)
            )
        return mapping

    def get_outputs(self, include_bytecode=True):
        return sorted(
            set(super().get_outputs(include_bytecode)) | set(self.get_output_mapping())
        )

    def run(self):
        super().run()
        for target, source in self.get_output_mapping().items():
            if Path(source).parent == Path("configs"):
                self.mkpath(str(Path(target).parent))
                self.copy_file(source, target)


setup(cmdclass={"build_py": BuildPy})
