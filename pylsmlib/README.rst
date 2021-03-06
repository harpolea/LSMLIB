========
PyLSMLIB
========

PyLSMLIB_ is a Python package that wraps the LSMLIB_ level set library
using Cython_. It arose out of the need to replace the rudimentary
level set capability in FiPy_ with a faster C-based level set library.
The interface is designed to be pythonic and match the interface to
Scikit-fmm_.

Hosted
======

<https://github.com/ktchu/LSMLIB.git/pylsmlib>

PyLSMLIB_ is hosted at Github_ in a subdirectory of LSMLIB_.

Requirements
============

The main requirements are working versions of LSMLIB_ and
Numpy_. Cython_ should not be a requirement as the generated ``*.c``
files are included with the distribution. See `requirements.txt`_ for
specific versions of dependencies used to run this package on the
maintainer's system. The `requirements.txt`_ file is auto-generated so
most of the packages listed are not necessarily required. However, if
you are having issues with installation and missing packages this
would be a good place to start looking.

Installation
============

The following should get you most of the way to an install.

::

    $ pip install numpy

Clone the git repository. To clone the original LSMLIB:

::

    $ git clone git://github.com/ktchu/LSMLIB.git LSMLIB

Or to clone this version:

::

   $ git clone git://github.com/harpolea/LSMLIB.git LSMLIB

See the Github_ project page for further details. After cloning,
install LSMLIB_ (consult `LSMLIB's install`_ if you have issues).

::

    $ cd .../LSMLIB
    $ mkdir build
    $ cd build
    $ ../configure
    $ make
    $ sudo make install

To install PyLSMLIB_.

::

    $ cd .../LSMLIB/pylsmlib
    $ python setup.py install

To recompile after changing Cython code

::

   $ cd .../LSMLIB/pylsmlib
   $ python setup.py build_ext --inplace

If have altered ``__init__.py``, may need to copy  from
``.../LSMLIB/pylsmlib/build/lib.linux-x86_64-2.7/pylsmlib`` to ``.../LSMLIB/pylsmlib/pylsmlib``, i.e.

::

   $ cd .../LSMLIB/pylsmlib
   $ cp build/lib.linux-x86_64-2.7/pylsmlib/__init__.py pylsmlib/

May need to copy files across to
``/home/alice/anaconda/lib/python2.7/site-packages/pylsmlib-0.1-py2.7-linux-x86_64.egg/pylsmlib/`` if want to run package from outside the ``LSMLIB`` folder (e.g. if calling it from in ``pyro``):

::

  $ cd ../LSMLIB/pylsmlib/pylsmlib
  $ cp __init__.py __init__.pyc lsmlib.so pythonisedfns.py 
  /home/alice/anaconda/lib/python2.7/site-packages/pylsmlib-0.1-py2.7-linux-x86_64.egg/pylsmlib/


Testing
=======

To run the tests

::

    >>> import pylsmlib
    >>> pylsmlib.test()

Documentation
=============

To generate the PyLSMLIB_ documentation as html:

::

    $ cd .../LSMLIB/pylsmlib/doc
    $ make html

To generate as a pdf:

::

   $ cd .../LSMLIB/pylsmlib/doc
   $ make latexpdf

The pdf can then be found in ``.../LSMLIB/pylsmlib/doc/_build/latex``.

If have added new functions to the Cython wrapper, don't forget to add them to ``.../LSMLIB/pylsmlib/doc/index.rst`` so that they are added to the documentation.

.. _LSMLIB: http://ktchu.serendipityresearch.org/software/lsmlib/index.html
.. _PyLSMLIB: https://github.com/ktchu/LSMLIB/tree/master/pylsmlib
.. _Github: https://github.com/ktchu/LSMLIB
.. _requirements.txt: https://github.com/ktchu/LSMLIB/blob/master/pylsmlib/requirements.txt
.. _Cython: http://cython.org/
.. _FiPy: http://www.ctcms.nist.gov/fipy/
.. _Scikit-fmm: http://packages.python.org/scikit-fmm/
.. _Numpy: http://numpy.scipy.org/
.. _LSMLIB's install: https://github.com/ktchu/LSMLIB/blob/master/INSTALL
