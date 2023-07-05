# lfdfiles/setup.py

"""Lfdfiles package Setuptools script."""

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


with open('lfdfiles/lfdfiles.py', encoding='utf-8') as fh:
    code = fh.read()

version = search(r"__version__ = '(.*?)'", code).replace('.x.x', '.dev0')
version += ('.' + buildnumber) if buildnumber else ''

description = search(r'"""(.*)\.(?:\r\n|\r|\n)', code)

readme = search(
    r'(?:\r\n|\r|\n){2}"""(.*)"""(?:\r\n|\r|\n){2}from __future__',
    code,
    re.MULTILINE | re.DOTALL,
)
readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

if 'sdist' in sys.argv:
    # update README, LICENSE, and CHANGES files

    with open('README.rst', 'w', encoding='utf-8') as fh:
        fh.write(readme)

    license = search(
        r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""',
        code,
        re.MULTILINE | re.DOTALL,
    )
    license = license.replace('# ', '').replace('#', '')

    with open('LICENSE', 'w', encoding='utf-8') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)

    revisions = search(
        r'(?:\r\n|\r|\n){2}(Revisions.*)- \.\.\.',
        readme,
        re.MULTILINE | re.DOTALL,
    ).strip()

    with open('CHANGES.rst', encoding='utf-8') as fh:
        old = fh.read()

    d = revisions.splitlines()[-1]
    old = old.split(d)[-1]
    with open('CHANGES.rst', 'w', encoding='utf-8') as fh:
        fh.write(revisions.strip())
        fh.write(old)


class build_ext(_build_ext):
    """Delay import numpy until build."""

    def finalize_options(self):
        _build_ext.finalize_options(self)
        if isinstance(__builtins__, dict):
            __builtins__['__NUMPY_SETUP__'] = False
        else:
            setattr(__builtins__, '__NUMPY_SETUP__', False)
        import numpy

        self.include_dirs.append(numpy.get_include())


# Work around "Cython in setup_requires doesn't work"
# https://github.com/pypa/setuptools/issues/1317
try:
    import Cython  # noqa

    ext = '.pyx'
except ImportError:
    ext = '.c'

if sys.platform == 'win32':
    extra_compile_args = ['/openmp']
    extra_link_args: list[str] = []
elif sys.platform == 'darwin':
    # https://mac.r-project.org/openmp/
    extra_compile_args = ['-Xclang', '-fopenmp']
    extra_link_args = ['-lomp']
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        'lfdfiles._lfdfiles',
        ['lfdfiles/_lfdfiles' + ext],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='lfdfiles',
    version=version,
    license='BSD',
    description=description,
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Christoph Gohlke',
    author_email='cgohlke@cgohlke.com',
    url='https://www.cgohlke.com',
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/lfdfiles/issues',
        'Source Code': 'https://github.com/cgohlke/lfdfiles',
        # 'Documentation': 'https://',
    },
    python_requires='>=3.9',
    install_requires=['numpy', 'tifffile', 'click'],
    setup_requires=['setuptools', 'numpy'],
    extras_require={
        'all': [
            'imagecodecs',
            'matplotlib',
            'czifile',
            'oiffile',
            'netpbmfile',
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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
