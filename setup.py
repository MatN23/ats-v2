from pathlib import Path

from setuptools import find_packages, setup

_HERE = Path(__file__).parent


def _read_requirements() -> list:
    req_path = _HERE / "requirements.txt"
    with open(req_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _read_long_description() -> str:
    readme_path = _HERE / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


setup(
    name="ats-v2",
    version="2.0.0",
    description="Adaptive Training System v2: a config-driven LLM training framework on DeepSpeed.",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    author="ats-v2 contributors",
    license="Apache-2.0",
    packages=find_packages(include=["ats", "ats.*"]),
    py_modules=["train", "evaluate", "export"],
    python_requires=">=3.10",
    install_requires=_read_requirements(),
    entry_points={
        "console_scripts": [
            "ats-train=train:main",
            "ats-eval=evaluate:main",
            "ats-export=export:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
