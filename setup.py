#!/usr/bin/env python

import pathlib

import pkg_resources
from setuptools import find_packages, setup

# from distutils.core import Extension, find_packages, setup



# bktree = Extension("BKTree",
#                    ["qa/tools/bktree/library.cc", "qa/tools/bktree/bktree.cc"],
#                    libraries=['boost_python3'],
#                    extra_compile_args=['-std=c++11'])
with pathlib.Path("requirements_deploy.txt").open() as f:
    install_requires = [
        str(requirement) for requirement in pkg_resources.parse_requirements(f)
    ]

setup(
    name='PetPedia',
    version='0.0.1',
    author='Jiang',
    # ext_modules=[],
    #   script_args=['build_ext'],
    # options={'build_ext': {
    #     'inplace': True
    # }},
    install_requires=install_requires,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    # package_dir={"": "qa"},
    # packages=find_packages(where="qa"),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "convert_model = deployment.convert:main",
        ],
    })
