# lfdfiles/setup.py

"""Lfdfiles package setuptools script."""

import sys
import re

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

buildnumber = ''


def search(pattern, code, flags=0):
    # return first match for pattern in code
    match = re.search(pattern, code, flags)
    if match is None:
        raise ValueError(f'{pattern!r} not found')
    return match.groups()[0]


with open('lfdfiles/lfdfiles.py') as fh:
    code = fh.read()

version = search(r"__version__ = '(.*?)'", code)
version += ('.' + buildnumber) if buildnumber else ''

description = search(r'"""(.*)\.(?:\r\n|\r|\n)', code)

readme = search(
    r'(?:\r\n|\r|\n){2}"""(.*)"""(?:\r\n|\r|\n){2}[__version__|from]',
    code,
    re.MULTILINE | re.DOTALL,
)
readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

if 'sdist' in sys.argv:
    # update README, LICENSE, and CHANGES files

    with open('README.rst', 'w') as fh:
        fh.write(readme)

    license = search(
        r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""',
        code,
        re.MULTILINE | re.DOTALL,
    )
    license = license.replace('# ', '').replace('#', '')

    with open('LICENSE', 'w') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)

    revisions = search(
        r'(?:\r\n|\r|\n){2}(Revisions.*)\* \.\.\.',
        readme,
        re.MULTILINE | re.DOTALL,
    ).strip()

    with open('CHANGES.rst', 'r') as fh:
        old = fh.read()

    d = revisions.splitlines()[-1]
    old = old.split(d)[-1]
    with open('CHANGES.rst', 'w') as fh:
        fh.write(revisions.strip())
        fh.write(old)


class build_ext(_build_ext):
    """Delay import numpy until build."""

    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


# Work around "Cython in setup_requires doesn't work"
# https://github.com/pypa/setuptools/issues/1317
try:
    import Cython  # noqa

    ext = '.pyx'
except ImportError:
    ext = '.c'

ext_modules = [
    Extension(
        'lfdfiles._lfdfiles',
        ['lfdfiles/_lfdfiles' + ext],
        extra_compile_args=[
            '/openmp' if sys.platform == 'win32' else '-fopenmp'
        ],
        extra_link_args=[] if sys.platform == 'win32' else ['-fopenmp'],
    ),
]

setup(
    name='lfdfiles',
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@uci.edu',
    license='BSD',
    url='https://www.lfd.uci.edu/~gohlke/',
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/lfdfiles/issues',
        'Source Code': 'https://github.com/cgohlke/lfdfiles',
        # 'Documentation': 'https://',
    },
    python_requires='>=3.8',
    install_requires=['numpy>=1.19.2', 'tifffile>=2021.11.2', 'click'],
    setup_requires=['setuptools>=18.0', 'numpy>=1.19.2'],
    extras_require={
        'all': [
            'imagecodecs>=2022.2.22',
            'matplotlib>=3.4',
            'czifile>=2019.7.2',
            'oiffile>=2021.6.6',
            'netpbmfile>=2021.6.6',
        ]
    },
    tests_require=['pytest'],
    packages=['lfdfiles'],
    entry_points={
        'console_scripts': [
            'lfdfiles = lfdfiles.__main__:main',
            'fbd2b64 = lfdfiles.fbd2b64:main',
        ]
    },
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    platforms=['any'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
