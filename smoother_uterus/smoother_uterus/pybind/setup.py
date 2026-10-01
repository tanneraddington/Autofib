from setuptools import setup, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os


gtsam_include = [os.environ.get("GTSAM_INCLUDE_DIR", "/usr/local/include")]
gtsam_lib = [os.environ.get("GTSAM_LIB_DIR", "/usr/local/lib")]


extra_compile_args = ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC"]

ext_modules = [
    Pybind11Extension(
        "smoother_uterus",
        sources=[
            "src/bindings.cpp",
            "src/HyperCalibrator.cpp",
            "../src/solver/SmootherSolver.cpp",
            "../src/solver/PositionPriorFactor.cpp",
            "../src/solver/CosseratTwistFactor.cpp",
            "../src/solver/TipProjectionFactor.cpp",
            "../src/solver/ConstantVelocityFactor.cpp",
        ],
        include_dirs=[
            *gtsam_include,
            "../src",
            '/usr/include/eigen3',
            pybind11.get_include()
        ],
        library_dirs=[
            *gtsam_lib,
        ],
        libraries=[
            "gtsam",
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="smoother_uterus",
    version="0.1",
    packages=find_packages(),
    author="Your Name",
    description="pybind11 wrapper for smoother_uterus",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
