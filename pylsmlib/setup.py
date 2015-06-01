#!/usr/bin/env python

r"""
Python interface for LSMLIB.

Python interface for LSMLIB using Cython.
"""

DOCLINES = __doc__.split("\n")

#import ez_setup
#ez_setup.use_setuptools()

#from setuptools import setup
#from setuptools.extension import Extension
from distutils.core import setup
from distutils.extension import Extension
import os
import numpy

#try:
#    from Cython.Distutils import build_ext
#except ImportError:
#    sources = [os.path.join('pylsmlib', 'lsmlib.c')]
#    cmdclass = {}
#else:
#    sources = [os.path.join('pylsmlib', 'lsmlib.pyx')]
#    cmdclass = { 'build_ext': build_ext }

from Cython.Distutils import build_ext
sources = [os.path.join('pylsmlib', 'lsmlib.pyx')]
cmdclass = { 'build_ext': build_ext }

setup(name="pylsmlib",
      version="0.1",
      author="Daniel Wheeler",
      author_email="daniel.wheeler2@gmail.com",
      description=DOCLINES[0],
      url='https://github.com/harpolea/LSMLIB.git',
      long_description = "\n".join(DOCLINES[2:]),
      license='BSD-style',
      packages=['pylsmlib'],
      cmdclass = cmdclass,
      ext_modules = [Extension("pylsmlib.lsmlib",
                                sources=sources,
                                extra_link_args=['-fPIC'],
                                libraries=['lsm_serial', 'lsm_toolbox'],
                                include_dirs=[numpy.get_include(), 'pylsmlib'])]
)
