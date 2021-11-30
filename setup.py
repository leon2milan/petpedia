#!/usr/bin/env python

from distutils.core import setup, Extension

bktree = Extension("BKTree",
                   ["qa/tools/bktree/library.cc", "qa/tools/bktree/bktree.cc"],
                   libraries=['boost_python3'],
                   extra_compile_args=['-std=c++11'])

setup(name='PetPedia',
      version='0.0.1',
      author='Jiang',
      ext_modules=[bktree],
      script_args=['build_ext'],
      options={'build_ext': {
          'inplace': True
      }},
      python_requires=">=3.6")