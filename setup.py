"""
setup.py \n
Enable installing through pip (preferably in editable mode)
"""

from setuptools import find_packages, setup

setup(
    name="option-critic",
    version="0.1.0",
    author="Ashrith Sagar",
    description="The Option-Critic Architecture",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["oca", "oca.*"]),
    package_dir={"": "."},
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": ["oca=oca.run:main"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.9",
)
