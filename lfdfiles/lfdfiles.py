# lfdfiles.py

# Copyright (c) 2012-2025, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Laboratory for Fluorescence Dynamics (LFD) file formats.

Lfdfiles is a Python library and console script for reading, writing,
converting, and viewing many of the proprietary file formats used
to store experimental data and metadata at the
`Laboratory for Fluorescence Dynamics <https://www.lfd.uci.edu/>`_.
For example:

- SimFCS VPL, VPP, JRN, BIN, INT, CYL, REF, BH, BHZ, B64, I64, Z64, R64
- FLIMbox FBD, FBF, FBS.XML
- GLOBALS LIF, ASCII
- CCP4 MAP
- Vaa3D RAW
- Bio-Rad(r) PIC
- ISS Vista IFLI, IFI
- FlimFast FLIF

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2025.5.10
:DOI: `10.5281/zenodo.8384166 <https://doi.org/10.5281/zenodo.8384166>`_

Quickstart
----------

Install the lfdfiles package and all dependencies from the
`Python Package Index <https://pypi.org/project/lfdfiles/>`_::

    python -m pip install -U "lfdfiles[all]"

Print the console script usage::

    python -m lfdfiles --help

The lfdfiles library is type annotated and documented via docstrings.

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/lfdfiles>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.10.11, 3.11.9, 3.12.10, 3.13.3 64-bit
- `Cython <https://pypi.org/project/cython/>`_ 3.1.0 (build)
- `NumPy <https://pypi.org/project/numpy/>`_ 2.2.5
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2025.5.10 (optional)
- `Czifile <https://pypi.org/project/czifile/>`_ 2019.7.2.1 (optional)
- `Oiffile <https://pypi.org/project/oiffile/>`_ 2025.1.1 (optional)
- `Netpbmfile <https://pypi.org/project/netpbmfile/>`_ 2025.5.8 (optional)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.10.3
  (optional, for plotting)
- `Click <https://pypi.python.org/pypi/click>`_ 8.1.8
  (optional, for command line apps)

Revisions
---------

2025.5.10

- Mark Cython extension free-threading compatible.
- Remove doctest command line option.
- Support Python 3.14.

2025.3.16

- Replace deprecated tifffile.stripnull function.
- Fix misspelled VistaIfli.header keys.
- Drop support for Python 3.9.

2024.10.24

- Fix variable length little-endian base 128 decoding.

2024.9.15

- Improve typing.
- Deprecate Python 3.9, support Python 3.13.

2024.5.24

- Fix docstring examples not correctly rendered on GitHub.

2024.4.24

- Support NumPy 2.

2024.3.4

- Fix decoding 32-bit, 16 windows, 4 channels Spartan6 FBD files (#1).

2023.9.26

- Remove phasor and lifetime methods from VistaIfli (breaking).
- Rename SimfcsFbd and SimfcsFbf to FlimboxFbd and FlimboxFbf (breaking).
- Deprecate SimfcsFbd and SimfcsFbf.
- Support int16 FLIMbox cross correlation phase indices (bins).
- Add FlimboxFbs class for ISS VistaVision FLIMbox settings.
- Add decoder for 32-bit, 16 windows, 4 channels FlimboxFbd (untested).

2023.9.16

- Rewrite VistaIfli based on file format specification (breaking).
- Define positional and keyword parameters (breaking).
- SimfcsFbd.asarray returns bins only (breaking).

2023.8.30

- …

Refer to the CHANGES file for older revisions.

Notes
-----

The API is not stable yet and might change between revisions.

Python <= 3.9 is no longer supported. 32-bit versions are deprecated.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2015-2022
<https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist>`_
is required on Windows.

Many of the LFD's file formats are not documented and might change arbitrarily.
This implementation is mostly based on reverse engineering existing files.
No guarantee can be made as to the correctness of code and documentation.

Experimental data are often stored in plain binary files with metadata
available in separate, human readable journal files (`.jrn`).

Unless specified otherwise, data are stored in little-endian, C contiguous
order.

References
----------

The following software is referenced in this module:

1.  `SimFCS <https://www.lfd.uci.edu/globals/>`_, a.k.a. Globals for
    Images, is software for fluorescence image acquisition, analysis, and
    simulation, developed by Enrico Gratton at UCI.
2.  `Globals <https://www.lfd.uci.edu/globals/>`_, a.k.a. Globals for
    Spectroscopy, is software for the analysis of multiple files from
    fluorescence spectroscopy, developed by Enrico Gratton at UIUC and UCI.
3.  ImObj is software for image analysis, developed by LFD at UIUC.
    Implemented on Win16.
4.  `FlimFast <https://www.cgohlke.com/flimfast/>`_ is software for
    frequency-domain, full-field, fluorescence lifetime imaging at video
    rate, developed by Christoph Gohlke at UIUC.
5.  FLImage is software for frequency-domain, full-field, fluorescence
    lifetime imaging, developed by Christoph Gohlke at UIUC.
    Implemented in LabVIEW.
6.  FLIez is software for frequency-domain, full-field, fluorescence
    lifetime imaging, developed by Glen Redford at UIUC.
7.  Flie is software for frequency-domain, full-field, fluorescence
    lifetime imaging, developed by Peter Schneider at MPIBPC.
    Implemented on a Sun UltraSPARC.
8.  FLOP is software for frequency-domain, cuvette, fluorescence lifetime
    measurements, developed by Christoph Gohlke at MPIBPC.
    Implemented in LabVIEW.
9.  `VistaVision <http://www.iss.com/microscopy/software/vistavision.html>`_
    is commercial software for instrument control, data acquisition and data
    processing by ISS Inc (Champaign, IL).
10. `Vaa3D <https://github.com/Vaa3D>`_ is software for multi-dimensional
    data visualization and analysis, developed by the Hanchuan Peng group at
    the Allen Institute.
11. `Voxx <https://voxx.sitehost.iu.edu/>`_ is a volume rendering program
    for 3D microscopy, developed by Jeff Clendenon et al. at the Indiana
    University.
12. `CCP4 <https://www.ccp4.ac.uk/>`_, the Collaborative Computational Project
    No. 4, is software for macromolecular X-Ray crystallography.

Examples
--------

Create a Bio-Rad PIC file from a NumPy array:

>>> data = numpy.arange(1000000).reshape(100, 100, 100).astype('u1')
>>> bioradpic_write('_biorad.pic', data)

Read the volume data from the PIC file as NumPy array, and access metadata:

>>> with BioradPic('_biorad.pic') as f:
...     f.shape
...     f.spacing
...     data = f.asarray()
...
(100, 100, 100)
(1.0, 1.0, 1.0)

Convert the PIC file to a compressed TIFF file:

>>> with BioradPic('_biorad.pic') as f:
...     f.totiff('_biorad.tif', compression='zlib')
...

"""

from __future__ import annotations

__version__ = '2025.5.10'

__all__ = [
    '__version__',
    'LfdFile',
    'LfdFileSequence',
    'LfdFileError',
    'RawPal',
    'SimfcsVpl',
    'SimfcsVpp',
    'SimfcsJrn',
    'SimfcsBin',
    'SimfcsRaw',
    'SimfcsInt',
    'SimfcsIntPhsMod',
    'SimfcsFit',
    'SimfcsCyl',
    'SimfcsRef',
    'SimfcsBh',
    'SimfcsBhz',
    'SimfcsB64',
    'SimfcsI64',
    'SimfcsZ64',
    'SimfcsR64',
    'SimfcsGpSeries',
    'FlimboxFbd',
    'FlimboxFbf',
    'FlimboxFbs',
    'GlobalsLif',
    'GlobalsAscii',
    'VistaIfli',
    'VistaIfi',
    'Ccp4Map',
    'Vaa3dRaw',
    'VoxxMap',
    'BioradPic',
    'FlimfastFlif',
    'FlimageBin',
    'FlieOut',
    'FliezI16',
    'FliezDb2',
    'NetpbmFile',
    'OifFile',
    'CziFile',
    'TiffFile',
    'convert_fbd2b64',
    'convert2tiff',
    'simfcsb64_write',
    'simfcsi64_write',
    'simfcsz64_write',
    'simfcsr64_write',
    'ccp4map_write',
    'vaa3draw_write',
    'voxxmap_write',
    'bioradpic_write',
    'flimbox_histogram',
    'flimbox_decode',
    'sflim_decode',
    # deprecated
    'SimfcsFbf',
    'SimfcsFbd',
    'simfcsfbd_histogram',
    'simfcsfbd_decode',
]

import copy
import logging
import math
import os
import re
import struct
import sys
import warnings
import zipfile
import zlib
from functools import cached_property
from typing import IO, TYPE_CHECKING

import numpy
import tifffile
from tifffile import (
    FileSequence,
    TiffWriter,
    askopenfilename,
    astype,
    imshow,
    parse_kwargs,
    product,
    update_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType
    from typing import Any, ClassVar, Literal, TypeVar

    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, DTypeLike, NDArray

# delay import optional modules
pyplot: ModuleType | None = None
cycler: ModuleType | None = None


def import_pyplot(fail: bool = True, /) -> bool:
    """Import `matplotlib.pyplot`."""
    global pyplot
    global cycler

    if pyplot is not None:
        return True
    try:
        import cycler as cyclr
        from matplotlib import pyplot as plt

        pyplot = plt
        cycler = cyclr
    except ImportError:
        if fail:
            raise
        return False
    return True


class LfdFileError(Exception):
    """Exception to indicate invalid LFD files.

    Parameters:
        arg:
            Error message or LfdFile instance where error occurred.
        msg:
            Additional error message.

    """

    def __init__(self, arg: LfdFile | str = '', /, msg: str = '') -> None:
        if isinstance(arg, LfdFile):
            arg = f'invalid {arg.__class__.__name__} file'
        if msg:
            arg = f'{arg}: {msg}' if arg else msg
        Exception.__init__(self, arg)


class LfdFileRegistry(type):
    """Metaclass to register classes derived from LfdFile."""

    classes: list[type[LfdFile]] = []
    """Registered LfdFile classes."""

    def __new__(
        mcs: Any, name: str, bases: tuple[type, ...], dct: dict[str, Any], /
    ) -> LfdFileRegistry:
        cls = type.__new__(mcs, name, bases, dct)
        if cls.__name__[:7] != 'LfdFile':
            LfdFileRegistry.classes.append(cls)
        return cls  # type: ignore[no-any-return]

    @staticmethod
    def sort() -> None:
        """Move SimfcsBin class to end of list.

        :meta private:

        """
        lst = LfdFileRegistry.classes
        lst.append(lst.pop(lst.index(SimfcsBin)))


class LfdFile(metaclass=LfdFileRegistry):
    """Base class for reading LFD files.

    Open file(s) and read headers and metadata.

    Parameters:
        filename:
            Name of file to open.
        validate:
            If True, filename must match :py:attr:`LfdFile._filepattern`.
        components:
            If True, open all component files found.
        _offset:
            Initial position of file pointer.
        **kwargs:
            Arguments passed to :py:meth:`LfdFile._init`.

    Examples:
        >>> with LfdFile('flimfast.flif') as f:
        ...     type(f)
        ...
        <class '...FlimfastFlif'>
        >>> with LfdFile('simfcs.ref', validate=False) as f:
        ...     type(f)
        ...
        <class '...SimfcsRef'>
        >>> with LfdFile('simfcs.bin', shape=(-1, 256, 256), dtype='u2') as f:
        ...     type(f)
        ...
        <class '...SimfcsBin'>

    """

    _filemode: ClassVar[str] = 'rb'
    """File open mode.

    :meta public:

    """

    _fileencoding: str | None = None
    """Text file encoding mode.

    :meta public:

    """

    _filepattern: ClassVar[str] = r'.*'
    """Regular expression pattern matching valid file names.

    :meta public:

    """
    _filesizemin: ClassVar[int] = 16
    """Minimum file size.

    :meta public:

    """
    _noplot: ClassVar[bool] = False
    """No plot.

    :meta public:

    """
    _figureargs: ClassVar[dict[str, Any]] = {'figsize': (6, 8.5)}
    """Arguments passed to `pyplot.figure`.

    :meta public:

    """

    # instance attributes
    shape: tuple[int, ...] | None
    """Shape of data array contained in file."""

    dtype: numpy.dtype[Any] | None
    """Type of array data contained in file."""

    axes: str | None
    """Character codes of array axes."""

    components: list[tuple[str, LfdFile]]
    """Component file label and class."""

    _fh: Any
    _pos: int | None
    _offset: int | None
    _fsize: int | None
    _filename: str
    _filepath: str

    def __new__(
        cls,
        filename: os.PathLike[Any] | str,
        /,
        **kwargs: Any,
    ) -> LfdFile:
        """Return LfdFile derived class that can open filename."""
        if cls is not LfdFile:
            return object.__new__(cls)

        update_kwargs(kwargs, validate=True)
        kwargs2 = parse_kwargs(kwargs, registry=None, skip=None)
        validate = kwargs['validate']
        registry = kwargs2['registry']
        skip = kwargs2['skip']

        if registry is None:
            registry = LfdFileRegistry.classes
        if skip is None:
            skip = set()
            if not validate:
                # skip formats that are too generic
                skip.update((SimfcsBh, SimfcsCyl, FlimboxFbd, FliezI16))
        exceptions = []
        for lfdfile in registry:
            if lfdfile in skip:
                continue
            try:
                with lfdfile(filename, **kwargs):
                    pass
            except FileNotFoundError:
                raise
            except Exception as exc:
                if 'does not match' not in str(exc):
                    import traceback

                    exceptions.append(
                        f'\n{lfdfile.__name__}\n\n{traceback.format_exc()}'
                    )
                continue
            else:
                return super().__new__(lfdfile)  # type: ignore[no-any-return]
        raise LfdFileError(
            'failed to read file using any LfdFile class.\n'
            + '\n'.join(exceptions)
        )

    def __init__(
        self,
        filename: os.PathLike[Any] | str,
        /,
        *,
        _offset: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs2 = parse_kwargs(kwargs, validate=True, components=True)
        components = bool(kwargs2['components'])
        validate = bool(kwargs2['validate'])
        self.shape = None
        self.dtype = None
        self.axes = None
        self._pos = None
        self._fsize = None
        self._offset = _offset

        self._filepath, self._filename = os.path.split(os.fspath(filename))
        components_ = self._components() if components else []
        if validate:
            self._valid_name()
        if components_:
            self._fh = None
            # verify file name is a component
            for label, fname in components_:
                if fname.lower() == self._filename.lower():
                    break
            else:
                raise LfdFileError(self, 'not a component file')
            # try to open file using all registered classes
            component_list: list[tuple[str, LfdFile]] = []
            for label, fname in components_:
                fname = os.path.join(self._filepath, fname)
                try:
                    lfdfile = self.__class__(
                        fname,
                        validate=validate,
                        components=False,
                        _offset=_offset,
                    )
                except Exception:  # LfdFileError, FileNotFoundError
                    continue
                component_list.append((label, lfdfile))
            if not component_list:
                raise LfdFileError(self, 'no component files found')
            self.components = component_list
            if lfdfile.shape is None:
                raise LfdFileError(self, 'no shape')
            self.shape = (len(component_list), *lfdfile.shape)
            self.dtype = lfdfile.dtype
            if component_list[0][1].axes is not None:
                self.axes = 'S' + component_list[0][1].axes
        else:
            self.components = []
            self._fh = open(
                filename, self._filemode, encoding=self._fileencoding
            )
            self._fh.seek(_offset)
            try:
                if self._filesizemin != len(self._fh.read(self._filesizemin)):
                    raise LfdFileError(self, 'file is too small')
            except LfdFileError:
                self._fh.close()
                self._fh = None
                raise
            except Exception as exc:
                self._fh.close()
                self._fh = None
                raise LfdFileError(self, 'not a text file') from exc

            self._fh.seek(_offset)
            try:
                self._init(**kwargs)
            except Exception:
                if self._fh is not None:
                    self._fh.close()
                    self._fh = None
                raise
            if self._fh is not None:
                self._pos = self._fh.tell()

    def __repr__(self) -> str:
        offset = f' @{self._offset}' if self._offset else ''
        filename = os.path.split(os.path.normcase(self.filename))[-1]
        return f'<{self.__class__.__name__} {filename!r}{offset}>'

    def __str__(self) -> str:
        s = [repr(self)]
        if self.components:
            components = ', '.join(i for i, c in self.components)
            s.append(f'components: {components}')
        if self.axes is not None:
            s.append(f'axes: {self.axes}')
        if self.shape is not None:
            s.append(f'shape: {self.shape}')
        if self.dtype is not None:
            s.append(f'dtype: {numpy.dtype(self.dtype)}')
        _str = self._str()  # pylint: disable=assignment-from-no-return
        if _str:
            s.append(_str)
        return indent(*s)

    def __enter__(self: LfdFileType) -> LfdFileType:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close file handle.

        Call :py:meth:`LfdFile._close`.

        """
        try:
            self._close()
        except Exception:
            pass
        if self._fh:
            self._fh.close()
            self._fh = None
        for _, component in self.components:
            component.close()

    def asarray(self, *args: Any, **kwargs: Any) -> NDArray[Any]:
        """Return data in file(s) as NumPy array.

        Parameters:
            **kwargs: Arguments passed to :py:meth:`LfdFile._asarray`.

        """
        if self.components:
            data = [c.asarray(*args, **kwargs) for _, c in self.components]
            return numpy.array(data).squeeze()
        if self._fh is not None and self._pos is not None:
            self._fh.seek(self._pos)
        return self._asarray(*args, **kwargs)

    def totiff(
        self,
        filename: os.PathLike[Any] | str | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        """Write image(s) and metadata to TIFF file.

        Parameters:
            filename:
                Name of TIFF file to write.
            **kwargs:
                Arguments passed to :py:meth:`LfdFile._totiff`.

        """
        if filename is None:
            filename = os.path.join(self._filepath, self._filename) + '.tif'
        kwargs2 = parse_kwargs(
            kwargs, imagej=False, ome=None, bigtiff=False, byteorder='<'
        )
        update_kwargs(kwargs, photometric='minisblack', software='lfdfiles')
        with TiffWriter(filename, **kwargs2) as tif:
            if self.components:
                for label, component in self.components:
                    kwargs2 = parse_kwargs(kwargs, description=label)
                    component._totiff(tif, **kwargs2)
            else:
                self._totiff(tif, **kwargs)

    def show(self, **kwargs: Any) -> None:
        """Display data in matplotlib figure.

        Parameters:
            **kwargs: Arguments passed to :py:meth:`LfdFile._plot`.

        """
        if self._noplot:
            return
        import_pyplot()
        assert pyplot is not None
        figure = pyplot.figure(facecolor='w', **self._figureargs)
        try:
            figure.canvas.manager.window.title('LfdFiles - ' + self._filename)
        except Exception:
            pass
        self._plot(figure, **kwargs)
        pyplot.show()

    def _init(self, **kwargs: Any) -> None:
        """Validate file and read metadata.

        :meta public:

        """

    def _close(self) -> None:
        """Free any allocated resources.

        :meta public:

        """

    def _components(self) -> list[tuple[str, str]]:
        """Return possible names of component files.

        :meta public:

        """
        return []

    def _asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Read data from file and return as NumPy array.

        :meta public:

        """
        raise NotImplementedError

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image and metadata to TIFF file.

        :meta public:

        """
        raise NotImplementedError

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display data in matplotlib figure.

        :meta public:

        """
        assert pyplot is not None
        update_kwargs(kwargs, cmap='viridis')
        pyplot.subplots_adjust(bottom=0.07, top=0.93)
        try:
            data = self.asarray()
            if isinstance(data, tuple):  # type: ignore[unreachable]
                data = data[0]  # type: ignore[unreachable]
            ndim = data.ndim
        except Exception as exc:
            warnings.warn(str(exc))
            ndim = None
        if ndim == 1:
            # plot line and histogram
            ax = pyplot.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
            ax.set_title(self._filename)
            ax.plot(data[:4096])
            ax = pyplot.subplot2grid((3, 1), (2, 0))
            ax.set_title('Histogram')
            ax.hist(data)
        elif ndim == 2:
            # plot image and histogram
            ax = pyplot.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
            ax.set_title(self._filename)
            ax.imshow(data, **kwargs)
            ax = pyplot.subplot2grid((3, 1), (2, 0))
            ax.set_title('Histogram')
            if data.dtype.kind in 'iu':
                bins = max(4, data.max() - data.min())
            else:
                bins = 64
            hist, bins = numpy.histogram(data, bins=bins)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            ax.bar(center, hist, align='center', width=width)
        elif ndim == 3:
            # plot MIP and mean of images
            image = numpy.max(data, axis=0)
            mean = numpy.mean(data, axis=(1, 2))
            ax = pyplot.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
            ax.set_title(self._filename + ' (MIP)')
            ax.imshow(image, **kwargs)
            ax = pyplot.subplot2grid((3, 1), (2, 0))
            ax.set_title('Mean')
            ax.set_xlim([0, len(mean) - 1])
            ax.plot(mean)
        else:
            pyplot.title("don't know how to plot the data")

    def _str(self) -> str | None:
        """Return extra information about file.

        :meta public:

        """

    def _valid_name(self) -> None:
        """Raise LfdFileError if filename does not match `_filepattern`.

        Re :py:attr:`LfdFile._filepattern`.

        """
        if not self._filepattern or self._filepattern == r'.*':
            return
        if re.search(self._filepattern, self._filename, re.IGNORECASE) is None:
            raise LfdFileError(
                self,
                f'.\n    File name {self._filename!r}'
                f' does not match {self._filepattern!r}',
            )

    def _decompress_header(
        self, max_length: int, /, max_read: int = 256
    ) -> bytes:
        """Return first uncompressed bytes of Zlib compressed file."""
        data = self._fh.read(max_read)
        self._fh.seek(0)
        if not data.startswith(b'\x78\x9c'):
            raise LfdFileError(self, 'not a Zlib compressed file')
        decompressor = zlib.decompressobj()
        return decompressor.decompress(data, max_length)

    def _fstat(self) -> os.stat_result:
        """Return status of open file."""
        return os.fstat(self._fh.fileno())

    @property
    def _filesize(self) -> int:
        """File size in bytes."""
        if self._fsize is None:
            pos = self._fh.tell()
            self._fh.seek(0, 2)
            self._fsize = self._fh.tell()
            self._fh.seek(pos)
        return self._fsize

    @property
    def filename(self) -> str:
        """Name of file."""
        return os.path.join(self._filepath, self._filename)

    @property
    def size(self) -> int | None:
        """Number of elements in data array."""
        if self.shape is None:
            return None
        return product(self.shape)

    @property
    def ndim(self) -> int | None:
        """Number of dimensions in data array."""
        if self.shape is None:
            return None
        return len(self.shape)


if TYPE_CHECKING:
    LfdFileType = TypeVar('LfdFileType', bound=LfdFile)


class LfdFileSequence(FileSequence):
    r"""Series of LFD files.

    Parameters:
        files:
            Glob filename pattern or sequence of file names.
        imread:
            Class or function to read image array from single file.
        pattern:
            Regular expression pattern matching axes names and chunk indices
            in file names.
        container:
            Name or open instance of ZIP file in which files are stored.
        sort:
            Function to sort file names if `files` is a pattern.
            If False, disable sorting.

    Examples:
        >>> ims = LfdFileSequence(
        ...     'gpint/v*.int',
        ...     pattern=r'v(?P<Channel>\d)(?P<Image>\d*).int',
        ...     imread=SimfcsInt,
        ... )
        >>> ims.axes
        'CI'
        >>> data = ims.asarray()
        >>> data.shape
        (2, 135, 256, 256)
        >>> ims.close()

    """

    _readfunction: ClassVar[
        type[LfdFile] | Callable[..., NDArray[Any]] | None
    ] = None
    """Function or class to read image array from single file.

    :meta public:

    """

    _indexpattern: ClassVar[str | None] = None
    """Regex pattern for axes names and chunk indices in filenames.

    :meta public:

    """

    def __init__(
        self,
        files: str | os.PathLike[Any] | Sequence[str | os.PathLike[Any]],
        /,
        *,
        imread: Callable[..., NDArray[Any]] | type[LfdFile] | None = None,
        pattern: str | None = None,
        container: str | os.PathLike[Any] | None = None,
        sort: Callable[..., Any] | bool | None = None,
    ) -> None:
        if pattern is None:
            pattern = self._indexpattern
        if imread is None:
            imread = self._readfunction
        if imread is None:
            raise ValueError('imread function not specified')

        imread_func: Callable[..., NDArray[Any]]

        if isinstance(imread, type) and issubclass(imread, LfdFile):

            def imread_func(
                fname: str | os.PathLike[Any],
                /,
                lfdfile: type[LfdFile] = imread,  # type: ignore[assignment]
                **kwargs: Any,
            ) -> NDArray[Any]:
                with lfdfile(fname) as lfdf:
                    return lfdf.asarray(**kwargs)

        else:
            imread_func = imread  # type: ignore[assignment]
        super().__init__(
            imread_func, files, container=container, sort=sort, pattern=pattern
        )


class RawPal(LfdFile):
    """Raw color palette.

    PAL files contain a single RGB or RGBA color palette, stored as 256x3 or
    256x4 unsigned bytes in C or Fortran order, without any header.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with RawPal('rgb.pal') as f:
        ...     print(f.asarray()[100])
        ...
        [ 16 255 239]
        >>> with RawPal('rgba.pal') as f:
        ...     print(f.asarray()[100])
        ...
        [219 253 187 255]
        >>> with RawPal('rrggbb.pal') as f:
        ...     print(f.asarray()[100])
        ...
        [182 114  91]
        >>> with RawPal('rrggbbaa.pal') as f:
        ...     print(f.asarray()[100])
        ...
        [182 114  91 170]
        >>> with RawPal('rrggbbaa.pal') as f:
        ...     print(f.asarray(order='F')[100])
        ...
        [182 114  91 170]

    """

    _filepattern = r'.*\.(pal|raw|bin|lut)$'
    _figureargs = {'figsize': (6, 1)}

    def _init(self, **kwargs: Any) -> None:
        """Verify file size is 768 or 1024."""
        if self._filesize not in {768, 1024}:
            raise LfdFileError(self)
        self.shape = 256, -1
        self.dtype = numpy.dtype(numpy.uint8)
        self.axes = 'XS'

    def _asarray(
        self, *, order: Literal['C', 'F'] | None = None, **kwargs: Any
    ) -> NDArray[numpy.uint8]:
        """Return palette data as uint8 array of shape (256, 3 or 4).

        Parameters:
            order:
                Determines whether the data is stored in C (row-major) or
                Fortran (column-major) order. By default the order is
                determined based on size and sum of differences.

        """
        assert self._fh is not None
        data = numpy.fromfile(self._fh, numpy.uint8).reshape(256, -1)
        if order is None:
            a = data.astype(numpy.int32)
            b = a.reshape(-1, 256).T
            if numpy.sum(numpy.abs(numpy.diff(a, axis=0))) > numpy.sum(
                numpy.abs(numpy.diff(b, axis=0))
            ):
                data = data.reshape(-1, 256).T
        elif order == 'F':
            data = data.reshape(-1, 256).T
        elif order != 'C':
            raise ValueError('unknown order', order)
        if data.shape[1] == 4 and numpy.all(data[:, 3] == 0):
            data[:, 3] = 255  # fix transparency
        return data

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display palette stored in file."""
        pal = self.asarray().reshape(1, 256, -1)
        ax = figure.add_subplot(1, 1, 1)
        ax.set_title(self._filename)
        ax.yaxis.set_visible(False)
        ax.imshow(pal, aspect=20, origin='lower', interpolation='nearest')

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write palette to TIFF file."""
        kwargs.update(photometric='rgb', planarconfig='contig')
        kwargs2 = parse_kwargs(kwargs, 'order')
        data = self.asarray(**kwargs2)
        data = numpy.expand_dims(data, axis=0)
        tif.write(data, **kwargs)


class SimfcsVpl(LfdFile):
    """SimFCS or ImObj color palette.

    SimFCS VPL files contain a single RGB color palette, stored as 256x3
    unsigned bytes in C or Fortran order, preceded by a 22- or 24-bytes header.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with SimfcsVpl('simfcs.vpl') as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs.vpl.tif')
        ...     print(f.shape, data[100])
        ...
        (256, 3) [189 210 246]
        >>> with TiffFile('_simfcs.vpl.tif') as f:
        ...     assert_array_equal(f.asarray()[0], data)
        ...

        >>> with SimfcsVpl('imobj.vpl') as f:
        ...     data = f.asarray()
        ...     f.totiff('_imobj.vpl.tif')
        ...     print(f.shape, data[100])
        ...
        (256, 3) [  0 254  27]
        >>> with TiffFile('_imobj.vpl.tif') as f:
        ...     assert_array_equal(f.asarray()[0], data)
        ...

    """

    _filepattern = r'.*\.vpl$'
    _figureargs = {'figsize': (6, 1)}

    name: str | None
    """Name of palette."""

    def _init(self, **kwargs: Any) -> None:
        """Verify file size and header."""
        assert self._fh is not None
        if self._filesize == 792:
            self._fh.seek(24)
            self.name = None
        elif self._filesize >= 790:
            if self._fh.read(7)[:6] != b'vimage':
                raise LfdFileError(self)
            self.name = bytes2str(self._fh.read(15))
        else:
            raise LfdFileError(self)
        self.shape = 256, 3
        self.dtype = numpy.dtype(numpy.uint8)
        self.axes = 'XS'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.uint8]:
        """Return palette data as uint8 array of shape (256, 3)."""
        assert self._fh is not None
        data = numpy.fromfile(self._fh, numpy.uint8, 768)
        if self._filesize == 792:
            return data.reshape(256, 3)
        return data.reshape(3, 256).T

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display palette stored in file."""
        pal = self.asarray().reshape(1, 256, -1)
        ax = figure.add_subplot(1, 1, 1)
        ax.set_title(self.name if self.name else self._filename)
        ax.yaxis.set_visible(False)
        ax.imshow(pal, aspect=20, origin='lower', interpolation='nearest')

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write palette to TIFF file."""
        kwargs.update(
            photometric='rgb', planarconfig='contig', description=self.name
        )
        data = numpy.expand_dims(self.asarray(), axis=0)
        tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return name of palette."""
        if self.name:
            return f'name: {self.name}'
        return None


class SimfcsVpp(LfdFile):
    """SimFCS color palettes.

    SimFCS VPP files contain multiple BGRA color palettes, each stored as
    256x4 values of unsigned bytes preceded by a 24-byte Pascal string.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with SimfcsVpp('simfcs.vpp') as f:
        ...     data = f.asarray('nice.vpl')
        ...     f.totiff('_simfcs.vpp.tif')
        ...     print(f.shape, data[100])
        ...
        (256, 4) [ 16 255 239 255]
        >>> with TiffFile('_simfcs.vpp.tif') as f:
        ...     assert_array_equal(f.asarray()[35, 0], data)
        ...

    """

    _filepattern = r'.*\.vpp$'
    _filesizemin = 24

    names: list[str]
    """Names of palettes in file."""

    def _init(self, **kwargs: Any) -> None:
        """Read list of palette names from file."""
        assert self._fh is not None
        self.names = []
        while True:
            try:
                # read Pascal string
                strlen = ord(self._fh.read(1))
                name = bytes2str(self._fh.read(23)[:strlen].lower())
            except Exception as exc:
                raise LfdFileError(self) from exc
            if not name.endswith('.vpl'):
                break
            self.names.append(name)
            self._fh.seek(1024, 1)
        if not self.names:
            raise LfdFileError(self)
        self.shape = 256, 4
        self.dtype = numpy.dtype(numpy.uint8)
        self.axes = 'XS'
        # TODO: do not assign to class variable
        self._figureargs = {  # type: ignore[misc]
            'figsize': (6, len(self.names) / 6)
        }

    def _asarray(
        self, key: str | int = 0, *, rgba: bool = True, **kwargs: Any
    ) -> NDArray[numpy.uint8]:
        """Return palette data uint8 array of shape (256, 4).

        Parameters:
            key:
                The index or name of the palette to return.
            rgba:
                If True, return RGBA palette, else BGRA.

        """
        assert self._fh is not None
        if isinstance(key, int):
            self.names[key]
        else:
            key = self.names.index(key)
        self._fh.seek(key * 1048 + 24)
        data = numpy.fromfile(self._fh, numpy.uint8, 1024).reshape(256, 4)
        if rgba:
            data[:, :3] = data[:, 2::-1]
        if numpy.all(data[:, 3] == 0):
            data[:, 3] = 255  # fix transparency
        return data

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display all palettes stored in file."""
        figure.subplots_adjust(top=0.96, bottom=0.02, left=0.18, right=0.95)
        for i, name in enumerate(self.names):
            a = self.asarray(i)
            a = a.reshape(1, 256, 4)
            ax = figure.add_subplot(len(self), 1, i + 1)
            if i == 0:
                ax.set_title(self._filename)
            ax.set_axis_off()
            ax.imshow(
                a, aspect='auto', origin='lower', interpolation='nearest'
            )
            pos = list(ax.get_position().bounds)
            figure.text(
                pos[0] - 0.01,
                pos[1],
                name[:-4],
                fontsize=10,
                horizontalalignment='right',
            )

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write all palettes to TIFF file."""
        kwargs.update(
            photometric='rgb',
            planarconfig='contig',
            contiguous=False,
            metadata=None,
        )
        for i, name in enumerate(self.names):
            data = self.asarray(key=i)
            data = numpy.expand_dims(data, axis=0)
            kwargs['description'] = name
            tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return names of all palettes in file."""
        return indent('names:', *self.names)

    def __len__(self) -> int:
        return len(self.names)


class SimfcsJrn(LfdFile):
    r"""SimFCS journal.

    SimFCS JRN files contain metadata for several measurements, stored as
    key, value pairs in an unstructured ASCII format. Records usually start
    with lines of 80 '\*' characters. The files do not contain array data.

    The metadata can be accessed as a list of dictionaries.

    Parameters:
        lower: Convert keys to lower case.

    Examples:
        >>> with SimfcsJrn('simfcs.jrn', lower=True) as f:
        ...     f[1]['paramters for tracking']['samplimg frequency']
        ...
        15625

    """

    _filemode = 'r'
    _fileencoding = 'cp1252'
    _filepattern = r'.*\.jrn$'
    _noplot = True

    # regular expressions of all keys found in journal files
    _keys_ = r"""
        Image experiment
        Correlation expt
        Card type
        Channel for tracking
        Up down flag
        DATE
        TIME
        Dark \d+
        Movement type
        Mode
        DC threshold
        EWmission
        Emission
        Extension
        Frames integrated
        Frequency domain
        Int scale factor\d+
        Linescan point\s*\d+
        Maximum cycles
        Points per orbit
        Sampli[mn]g frequency
        R harmonics
        R%
        Mod for auto R
        Points per pixel
        Radius
        Dwell time
        Period
        Rperiods
        Sampling freq
        Scan type
        Scanner t-constant
        Scanner time const
        Scanner voltage
        Scanning period
        Scanning radius
        Time/photon mode
        Number or r-periods
        Cycles per particle
        Voltage full scale
        DC threshold
        Z[_-]radius
        Z[_-]Period
        frame_[xy]_range
        wait_end_of_frame
        wait_end_of_line
        [xyz][0o]_frame
        [xyz]_Offset
        [xyz]_Position
        [xyz]_Range\d*
        [PM]\d_\d calibration factor for FLIMbox
        FLIM BOX data
        roi (?:serial|parrallel|parallel) (?:start|width|bin)
        clear cycles
        clearing mode
        amplifier gain
        on-chip multiplier gain
        pmode
        shutter open mode
        exposure time
        frame time
        total time
        frames written
        """

    _keys: re.Pattern[str] = re.compile(
        '({})'.format(
            '|'.join(
                i.strip().replace(' ', '[ ]')
                for i in _keys_.splitlines()
                if i.strip()
            )
        )
    )
    _skip = re.compile(r'\s*\(.*\)')  # ignore parenthesis in values

    _lower: bool
    _records: list[dict[str, str]]

    def _init(self, *, lower: bool = False, **kwargs: Any) -> None:
        """Read journal file and parse into list of dictionaries."""
        assert self._fh is not None
        firstline = self._fh.readline()
        if not firstline.startswith('*' * 80) or firstline.startswith('roi'):
            raise LfdFileError(self)
        content = self._fh.read()
        self._fsize = len(firstline) + len(content)
        if not firstline.startswith('***'):
            content = firstline + '\n' + content
        lower = bool(lower)
        self._lower = lower
        self._records = []
        for record in content.split('*' * 80):
            recdict = {}
            record = record.split('COMMENTS', 1)
            if len(record) > 1:
                record, comments = record
            else:
                record, comments = record[0], ''
            recdict['comments' if lower else 'COMMENTS'] = comments.strip(
                '* :;=\n\r\t'
            )
            record = re.split(r'[*]{5}([\w\s]+)[*]{5}', record)
            for key, value in zip(record[1::2], record[2::2]):
                newdict: dict[str, Any] = {}
                key = key.strip('* :;=\n\r\t')
                if lower:
                    key = key.lower()
                value = self._parse_journal(
                    value, self._keys, result=newdict, lower=lower
                )
                recdict[key] = newdict
            self._parse_journal(
                record[0], self._keys, result=recdict, lower=lower
            )
            self._records.append(recdict)
        self.close()

    def _asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Raise ValueError."""
        raise ValueError('SimfcsJrn file does not contain array data')

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        raise ValueError('SimfcsJrn file does not contain image data')

    @staticmethod
    def _parse_journal(
        journal: str,
        repattern: re.Pattern[str],
        result: dict[str, Any] | None,
        lower: bool,
    ) -> dict[str, Any]:
        """Return dictionary of keys and values in journal string."""
        if result is None:
            result = {}

        keyval = re.split(repattern, journal, maxsplit=0)
        keyval = list(s.strip('* :;=\n\r\t') for s in keyval)
        val = [astype(re.sub(SimfcsJrn._skip, '', v)) for v in keyval[2:-1:2]]
        key = [k.lower() if lower else k for k in keyval[1:-1:2]]
        result.update(zip(key, val))
        return result

    def _str(self) -> str | None:
        """Return string with information about file."""
        comments = 'comments' if self._lower else 'COMMENTS'
        return '\n'.join(
            indent(
                f'Record {i}',
                format_dict(record, prefix='', excludes=[comments, '_']),
                indent(comments, record[comments]),
            )
            for i, record in enumerate(self._records)
        )

    def __getitem__(self, key: int, /) -> dict[str, str]:
        """Return selected record."""
        return self._records[key]

    def __len__(self) -> int:
        """Return number of records."""
        return len(self._records)

    def __iter__(self) -> Iterator[dict[str, str]]:
        """Return iterator over records."""
        return iter(self._records)


class SimfcsBin(LfdFile):
    """SimFCS raw binary data.

    SimFCS BIN and RAW files contain homogeneous array data of any type and
    shape, stored C-contiguously in little-endian byte order.
    A common format is: shape=(256, 256), dtype='uint16'.

    Parameters:
        filename:
            Name of file to open.
        shape:
            Shape of array to read from file.
        dtype:
            Datatype of array in file.
        offset:
            Position in bytes of array data in file.
            Use to skip file header.
        validate_size:
            If True, file size must exactly match offset, data shape and
            dtype.

    Examples:
        >>> with SimfcsBin(
        ...     'simfcs.bin', shape=(-1, 256, 256), dtype='uint16'
        ... ) as f:
        ...     data = f.asarray(memmap=True)
        ...     f.totiff('_simfcs.bin.tif', compression='zlib')
        ...     print(f.shape, data[751, 127, 127])
        (752, 256, 256) 1
        >>> with TiffFile('_simfcs.bin.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(bin|raw)$'

    def _init(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        offset: int = 0,
        validate_size: bool = True,
        **kwargs: Any,
    ) -> None:
        """Validate file size is multiple of shape and dtype."""
        assert self._fh is not None
        if not 0 <= offset <= self._filesize:
            raise LfdFileError(self, 'offset out of range')
        shapes: list[tuple[int, ...]]
        dtypes: list[DTypeLike]
        if shape is None:
            shapes = [(256, 256), (-1, 256, 256), (128, 128), (64, 64)]
        else:
            shapes = [shape]
        if dtype is None:
            dtypes = [numpy.uint16, numpy.uint8]
        else:
            dtypes = [dtype]
        for shape, dtype in ((s, d) for s in shapes for d in dtypes):
            try:
                self.shape = determine_shape(
                    shape,
                    dtype,
                    self._filesize - offset,
                    validate=validate_size,
                )
                break
            except Exception:
                pass
        else:
            raise LfdFileError(self, 'shape and dtype do not match file size')
        self.dtype = numpy.dtype(dtype)
        self._fh.seek(offset)

    def _asarray(self, *, memmap: bool = False, **kwargs: Any) -> NDArray[Any]:
        """Return data as array of specified shape and type.

        Parameters:
            memmap: Return a read-only memory-map to the data array on disk.

        """
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        dtype = '<' + self.dtype.char
        if memmap:
            return numpy.memmap(
                self._fh,
                dtype,
                mode='r',
                offset=0 if self._pos is None else self._pos,
                shape=self.shape,
            )
        data = numpy.fromfile(self._fh, dtype, count=product(self.shape))
        return data.reshape(*self.shape)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        tif.write(self.asarray(), **kwargs)


SimfcsRaw = SimfcsBin


class SimfcsInt(LfdFile):
    """SimFCS intensity image.

    SimFCS INT files contain a single intensity image, stored as 256x256
    float32 or uint16 (older format). The measurement extension is usually
    encoded in the file name.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with SimfcsInt('simfcs2036.int') as f:
        ...     print(f.asarray()[255, 255])
        ...
        3.0
        >>> with SimfcsInt('simfcs1006.int') as f:
        ...     print(f.asarray()[255, 255])
        ...
        9

    """

    _filepattern = r'.*\.(int|ac)$'

    def _init(self, **kwargs: Any) -> None:
        """Validate file size is 256 KB."""
        if self._filesize == 262144:
            self.dtype = numpy.dtype('<f4')
        elif self._filesize == 131072:
            self.dtype = numpy.dtype('<u2')
        else:
            raise LfdFileError(self, 'file size mismatch')
        self.shape = 256, 256
        self.axes = 'YX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.uint16 | numpy.float32]:
        """Return data as 256x256 array of float32 or uint16."""
        assert self._fh is not None
        return numpy.fromfile(self._fh, self.dtype).reshape(256, 256)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        tif.write(self.asarray(), **kwargs)


class SimfcsIntPhsMod(LfdFile):
    """SimFCS lifetime component images.

    SimFCS INT, PHS and MOD files contain fluorescence lifetime image data
    from frequency-domain measurements.
    Three 256x256 float32 images are stored in separate files:
    intensity (``.int``), phase (``.phs``) and modulation (``.mod``).
    Phase values are in degrees, modulation in percent.
    The measurement extension and channel are often encoded in the file name.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with SimfcsIntPhsMod('simfcs_1000.phs') as f:
        ...     print(f.asarray().mean((1, 2)))
        ...
        [5.717 0 0.04645]

    """

    _filepattern = r'.*\.(int|phs|mod)$'
    _figureargs = {'figsize': (6, 8)}

    def _init(self, **kwargs: Any) -> None:
        """Validate file size is 256 KB."""
        if self._filesize != 262144:
            raise LfdFileError(self, 'file size mismatch')
        self.dtype = numpy.dtype('<f4')
        self.shape = 256, 256
        self.axes = 'YX'

    def _components(self) -> list[tuple[str, str]]:
        """Return possible names of component files."""
        return [(c, self._filename[:-3] + c) for c in ('int', 'phs', 'mod')]

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return image data as float32 array of shape (256, 256)."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        tif.write(self.asarray(), **kwargs)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in files."""
        assert pyplot is not None
        update_kwargs(kwargs, cmap='viridis')
        images = self.asarray()
        pyplot.subplots_adjust(bottom=0.03, top=0.97, hspace=0.1, wspace=0.1)
        axes = [
            pyplot.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2),
            pyplot.subplot2grid((3, 2), (2, 0)),
            pyplot.subplot2grid((3, 2), (2, 1)),
        ]
        for i, (img, ax, title) in enumerate(
            zip(images, axes, (self._filename + ' - int', 'phs', 'mod'))
        ):
            ax.set_title(title)
            if i == 0:
                ax.imshow(img, vmin=0, **kwargs)
            else:
                ax.set_axis_off()
                ax.imshow(img, **kwargs)


class SimfcsFit(LfdFile):
    """SimFCS fit data.

    SimFCS FIT files contain results from image scan analysis.
    The fit parameters are stored as a 1024x16 float64 array, followed by
    an 8 bytes buffer and the intensity image used for the fit, stored as
    a 256x256 float32 array.

    The 16 fit parameters are::

        W0
        Background
        Pixel size
        Triplet amplitude
        G1
        D1 (um2/s)
        G2
        D2 (um2/s)
        Exp aplitude
        Exp time/Ch1 int
        Triplet rate/Ch2 int
        Fraction vesicle
        Radius vesicle
        Velocity modulus
        Velocity x
        Velocity y

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with SimfcsFit('simfcs.fit') as f:
        ...     dc_ref = f.asarray()
        ...     p_fit = f.p_fit(size=7)
        ...     f.totiff('_simfcs.fit.tif')
        ...     print(f'{p_fit[6, 1, 1]:.3f} {dc_ref[128, 128]:.2f}')
        ...
        0.937 20.23
        >>> with TiffFile('_simfcs.fit.tif') as f:
        ...     assert_array_equal(f.asarray(), dc_ref)
        ...

    """

    _filepattern = r'.*\.fit$'

    # type of data in file
    _record_t = numpy.dtype(
        [
            ('p_fit', '<f8', (1024, 16)),
            ('_', '<f8'),
            ('dc_ref', '<f4', (256, 256)),
        ]
    )
    # 16 parameter labels
    _labels = (
        'W0',
        'Background',
        'Pixel size',
        'Triplet amplitude',
        'G1',
        'D1 (um2/s)',
        'G2',
        'D2 (um2/s)',
        'Exp aplitude',
        'Exp time/Ch1 int',
        'Triplet rate/Ch2 int',
        'Fraction vesicle',
        'Radius vesicle',
        'Velocity modulus',
        'Velocity x',
        'Velocity y',
    )

    def _init(self, **kwargs: Any) -> None:
        """Validate file size is 384 KB."""
        if self._filesize != 393224:
            raise LfdFileError(self, 'file size mismatch')

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return intensity image as NumPy arrays."""
        return self.dc_ref()

    def dc_ref(self) -> NDArray[numpy.float32]:
        """Return intensity image as NumPy arrays."""
        assert self._fh is not None
        self._fh.seek(131080)
        return numpy.fromfile(self._fh, '<f4', 65536).reshape(256, 256)

    def p_fit(self, size: int = 32) -> NDArray[numpy.float64]:
        """Return fit parameters as NumPy arrays.

        Parameters:
            size: Number of rows and columns of fit parameters array.

        """
        assert self._fh is not None
        if not 0 < size <= 32:
            raise ValueError('size out of range [1..32]')
        self._fh.seek(0)
        p_fit = numpy.fromfile(self._fh, '<f8', 16384).reshape(1024, 16)
        return p_fit[: size * size].reshape(size, size, 16)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image and fit data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        tif.write(self.dc_ref(), description='dc_ref', **kwargs)
        tif.write(self.p_fit(), description='p_fit', **kwargs)

    def _str(self) -> str | None:
        """Return additional information about file."""
        return 'dc_ref: (256, 256) float32\np_fit: (32, 32, 16) float64'


class SimfcsCyl(LfdFile):
    """SimFCS orbital tracking data.

    SimFCS CYL files contain intensity data from orbital tracking
    measurements, stored as a uint16 array of shape
    (2 channels, number of orbits, 256 points per orbit).

    The number of channels and points per orbit can be read from the
    associated journal file.

    Parameters:
        filename:
            Name of file to open.
        shape:
            Number of channels, orbits, and points per orbit.

    Examples:
        >>> with SimfcsCyl('simfcs.cyl') as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs.cyl.tif')
        ...     print(f.shape, data[0, -1, :4])
        ...
        (2, 3291, 256) [109 104 105 112]
        >>> with TiffFile('_simfcs.cyl.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.cyl$'
    _figureargs = {'figsize': (6, 3)}

    def _init(
        self, *, shape: tuple[int, int, int] = (2, -1, 256), **kwargs: Any
    ) -> None:
        """Verify file size matches shape."""
        channels, orbits, points_per_orbit = shape
        if channels > 2 or channels < 1:
            raise ValueError(f'{channels=} out of range [1..2]')
        if points_per_orbit > 256 or points_per_orbit < 1:
            raise ValueError(f'{points_per_orbit=} out of range [1..256]')
        if orbits <= 0:
            orbits = points_per_orbit * channels * 2
            if self._filesize % orbits:
                raise LfdFileError(self, 'invalid shape')
            orbits = int(self._filesize // orbits)
        elif self._filesize != points_per_orbit * orbits * channels * 2:
            raise LfdFileError(self, 'invalid shape')
        self.shape = channels, orbits, points_per_orbit
        self.dtype = numpy.dtype('<u2')

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.uint16]:
        """Return data as (channels, -1, points_per_orbit) array of uint16."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in file."""
        assert pyplot is not None
        update_kwargs(kwargs, cmap='viridis', vmin=0)
        pyplot.subplots_adjust(bottom=0.1, top=0.9, hspace=0.2, wspace=0.1)
        ch0, ch1 = self.asarray()[:2]
        ax = pyplot.subplot2grid((2, 1), (0, 0))
        ax.set_title(self._filename)
        ax.imshow(ch0.T, aspect='auto', **kwargs)
        ax.set_yticks([])  # 0, ch0.shape[1]])
        ax = pyplot.subplot2grid((2, 1), (1, 0), sharex=ax, sharey=ax)
        ax.imshow(ch1.T, aspect='auto', **kwargs)
        pyplot.setp(ax.get_xticklabels(), visible=False)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        for data in self.asarray():
            tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return additional information about file."""
        assert self.shape is not None
        return 'channels: {}\norbits: {}\npoints_per_orbit: {}'.format(
            *self.shape
        )


class SimfcsRef(LfdFile):
    """SimFCS referenced fluorescence lifetime images.

    SimFCS REF files contain referenced fluorescence lifetime image data.
    Five or more 256x256 float32 images are stored consecutively:

    0. dc - intensity
    1. ph1 - phase of 1st harmonic
    2. md1 - modulation of 1st harmonic
    3. ph2 - phase of 2nd harmonic
    4. md2 - modulation of 2nd harmonic

    Phase values are in degrees, the modulation values are normalized.
    Phase and modulation values may be NaN.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with SimfcsRef('simfcs.ref') as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs.ref.tif')
        ...     print(f.shape, data[:, 255, 255])
        ...
        (5, 256, 256) [301.3 44.71 0.6185 68.13 0.3174]
        >>> with TiffFile('_simfcs.ref.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.ref$'
    _figureargs = {'figsize': (6, 11)}

    def _init(self, **kwargs: Any) -> None:
        """Verify file size is 1280 KB."""
        if self._filesize != 1310720:
            raise LfdFileError(self)
        self.shape = 5, 256, 256
        self.dtype = numpy.dtype('<f4')
        self.axes = 'SYX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return images as float32 array of shape (-1, 256, 256)."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in file."""
        assert pyplot is not None
        update_kwargs(kwargs, cmap='viridis')
        images = self.asarray()
        pyplot.subplots_adjust(bottom=0.02, top=0.97, hspace=0.1, wspace=0.1)
        axes = [
            pyplot.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=2),
            pyplot.subplot2grid((4, 2), (2, 0)),
            pyplot.subplot2grid((4, 2), (2, 1)),
            pyplot.subplot2grid((4, 2), (3, 0)),
            pyplot.subplot2grid((4, 2), (3, 1)),
        ]
        for i, (img, ax, title) in enumerate(
            zip(images, axes, (self._filename, 'ph1', 'md1', 'ph2', 'md2'))
        ):
            ax.set_title(title)
            if i == 0:
                ax.imshow(img, vmin=0, **kwargs)
            else:
                ax.set_axis_off()
                ax.imshow(img, **kwargs)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=True)
        for image, label in zip(self.asarray(), 'dc ph1 md1 ph2 md2'.split()):
            tif.write(image, description=label, **kwargs)


class SimfcsBh(LfdFile):
    """SimFCS Becker and Hickl fluorescence lifetime histogram.

    SimFCS B&H files contain time-domain fluorescence lifetime histogram data,
    acquired from Becker and Hickl(r) TCSPC cards, or converted from other
    data sources.
    The data are stored consecutively as 256 bins of 256x256 float32 images.
    B&H files are occasionally used to store consecutive 256x256 float32
    images, for example, volume data.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with SimfcsBh('simfcs.b&h') as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs.b&h.tif')
        ...     print(f.shape, data[59, 1, 84])
        ...
        (256, 256, 256) 12.0
        >>> with TiffFile('_simfcs.b&h.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(b&h)$'

    def _init(self, **kwargs: Any) -> None:
        """Verify file size is multiple of 262144."""
        if self._filesize % 262144:
            raise LfdFileError(self)
        self.shape = int(self._filesize // 262144), 256, 256
        self.dtype = numpy.dtype('<f4')
        self.axes = 'QYX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return image as float32 array of shape (-1, 256, 256)."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, compression='zlib', contiguous=False)
        data = self.asarray()
        # reshape according to axes provided by user
        axes = parse_kwargs(kwargs, axes=None)['axes']
        if axes:
            shape = {'ZYX': (0, None, 1, 2), 'TYX': (0, None, None, 1, 2)}[
                axes
            ]
            data.shape = tuple(
                1 if i is None else data.shape[i] for i in shape
            )
        tif.write(data, **kwargs)


class SimfcsBhz(SimfcsBh):
    """SimFCS compressed Becker and Hickl fluorescence lifetime histogram.

    SimFCS BHZ files contain time-domain fluorescence lifetime histogram data,
    acquired from Becker and Hickl(r) TCSPC cards, or converted from other
    data sources.
    SimFCS BHZ files are zipped B&H files: a Zip archive containing a single
    B&H file.

    Examples:
        >>> with SimfcsBhz('simfcs.bhz') as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs.bhz.tif')
        ...     print(f.shape, data[59, 1, 84])
        ...
        (256, 256, 256) 12.0
        >>> with TiffFile('_simfcs.bhz.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(bhz)$'

    _fh: IO[bytes]

    def _init(self, **kwargs: Any) -> None:
        """Verify Zip file contains file with size multiple of 262144."""
        with zipfile.ZipFile(self._fh) as zf:
            try:
                filesize = zf.filelist[0].file_size
            except (zipfile.BadZipfile, IndexError, AttributeError) as exc:
                raise LfdFileError(self) from exc
        if filesize % 262144:
            raise LfdFileError(self)
        self.shape = filesize // 262144, 256, 256
        self.dtype = numpy.dtype('<f4')
        self.axes = 'QYX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return image as float32 array of shape (256, 256, 256)."""
        assert self._fh is not None
        assert self.shape is not None
        with zipfile.ZipFile(self._fh) as zf:
            data = zf.read(zf.filelist[0])
        return numpy.frombuffer(data, self.dtype).reshape(self.shape).copy()

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, compression='zlib', contiguous=False)
        tif.write(self.asarray(), **kwargs)


class SimfcsB64(LfdFile):
    """SimFCS integer intensity data.

    SimFCS B64 files contain one or more square intensity images, a carpet
    of lines, or a stream of intensity data.
    The intensity data are stored as int16 contiguously after one int32
    defining the image size in x and/or y dimensions if applicable.
    The measurement extension and 'carpet' identifier are usually encoded
    in the file name.

    Parameters:
        filename:
            Name of file to open.
        dtype:
            Type of data in file.
        maxsize:
            Maximum square image length.

    Examples:
        >>> with SimfcsB64('simfcs.b64') as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs.b64.tif', compression='zlib')
        ...     print(f.shape, data[101, 255, 255])
        ...
        (102, 256, 256) 0
        >>> with TiffFile('_simfcs.b64.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.b64$'

    def _init(
        self,
        *,
        dtype: DTypeLike = '<i2',
        maxsize: int = 4096,
        **kwargs: Any,
    ) -> None:
        """Read file header."""
        assert self._fh is not None
        size = struct.unpack('<i', self._fh.read(4))[0]
        if not 1 <= size <= maxsize:
            raise LfdFileError(self, 'image size out of range')
        self.isize = size
        self.shape = size, size
        self.dtype = numpy.dtype(dtype)
        self.axes = 'YX'
        # determine number of images in file
        size = product(self.shape) * self.dtype.itemsize
        fsize = self._filesize - 4
        if fsize % self.dtype.itemsize:
            raise ValueError('file size mismatch')
        if 'carpet' in self._filename.lower():
            self.shape = (
                int((fsize // self.dtype.itemsize) // self.isize),
                self.isize,
            )
        elif fsize % size:
            # data stream or carpet
            self.shape = (int(fsize // self.dtype.itemsize),)
            self.axes = 'X'
        elif fsize // size > 1:
            # multiple images
            self.shape = (int(fsize // size),) + self.shape
            self.axes = 'IYX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.int16]:
        """Return intensity data as 1D, 2D, or 3D array of int16."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        count = product(self.shape)
        data = numpy.fromfile(self._fh, '<' + self.dtype.char, count=count)
        return data.reshape(*self.shape)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        tif.write(self.asarray(), **kwargs)


def simfcsb64_write(
    filename: os.PathLike[Any] | str,
    data: ArrayLike,
    /,
) -> None:
    """Write array of square int16 images to B64 file.

    Refer to :py:class:`SimfcsB64` for the B64 file format.

    Parameters:
        filename:
            Name of file to write.
        data:
            Data to write to file.
            Must be of shape (-1, size, size) and type int16.

    Examples:
        >>> data = (
        ...     numpy.arange(5 * 256 * 256)
        ...     .reshape(5, 256, 256)
        ...     .astype('int16')
        ... )
        >>> simfcsb64_write('_test.b64', data)
        >>> with SimfcsB64('_test.b64') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """
    data = numpy.asarray(data)
    if data.dtype.char != 'h':
        raise ValueError(f'invalid data type {data.dtype} (must be int16)')
    # TODO: write carpet
    if data.ndim != 3 or data.shape[1] != data.shape[2]:
        raise ValueError(f'invalid shape {data.shape}')
    with open(filename, 'wb') as fh:
        fh.write(struct.pack('I', data.shape[-1]))
        data.tofile(fh)


class SimfcsI64(LfdFile):
    """SimFCS compressed intensity image.

    SimFCS I64 files contain a single square intensity image, stored as a
    Zlib deflate compressed stream of one int32 (defining the image size in
    x and y dimensions) and the float32 image data.
    The measurement extension is usually encoded in the file name.
    Update 2020: Sometimes multiple images are stored in I64 files.

    Parameters:
        filename:
            Name of file to open.
        dtype:
            Type of data in file.
        maxsize:
            Maximum square image length.

    Examples:
        >>> with SimfcsI64('simfcs1000.i64') as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs1000.i64.tif')
        ...     print(f.shape, data[128, 128])
        ...
        (256, 256) 12.3125
        >>> with TiffFile('_simfcs1000.i64.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.i64$'

    def _init(
        self,
        *,
        dtype: DTypeLike = '<f4',
        maxsize: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Read file header."""
        if not 32 <= self._filesize <= 67108864:  # limit to 64 MB
            raise LfdFileError(self, 'file size out of range')
        size = struct.unpack('<i', self._decompress_header(4))[0]
        if not 2 <= size <= maxsize:
            raise LfdFileError(self, 'image size out of range')
        self.shape = size, size
        self.dtype = numpy.dtype(dtype)
        self.axes = 'YX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return data as 2D array of float32."""
        assert (
            self._fh is not None
            and self.shape is not None
            and self.dtype is not None
        )
        bufsize = product(self.shape) * self.dtype.itemsize + 4
        rawdata = zlib.decompress(self._fh.read(), 15, bufsize)
        data = numpy.frombuffer(rawdata, self.dtype, offset=4)
        data = data.copy()  # make writable
        try:
            data.shape = self.shape[0], self.shape[1]
        except Exception:
            # Z64 format with I64 extension
            data.shape = -1, self.shape[0], self.shape[1]
        return data

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        tif.write(self.asarray(), **kwargs)


def simfcsi64_write(
    filename: os.PathLike[Any] | str,
    data: ArrayLike,
    /,
) -> None:
    """Write a single float32 image to I64 file.

    Refer to :py:class:`SimfcsI64` for the I64 file format.

    Parameters:
        filename:
            Name of file to write.
        data:
            Data to write. Must be of shape (size, size) and type float32.

    Examples:
        >>> data = numpy.arange(256 * 256).reshape(256, 256).astype('float32')
        >>> simfcsi64_write('_test.i64', data)
        >>> with SimfcsI64('_test.i64') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """
    data = numpy.asarray(data)
    if data.dtype.char != 'f':
        raise ValueError(f'invalid data type {data.dtype} (must be float32)')
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f'invalid shape {data.shape}')
    rawdata = struct.pack('I', data.shape[0]) + data.tobytes()
    rawdata = zlib.compress(rawdata)
    with open(filename, 'wb') as fh:
        fh.write(rawdata)


class SimfcsZ64(LfdFile):
    """SimFCS compressed image stack.

    SimFCS Z64 files contain stacks of square images such as intensity volumes
    or time-domain fluorescence lifetime histograms acquired from
    Becker and Hickl(r) TCSPC cards.
    The data are stored as Zlib deflate compressed stream of two int32
    (defining the image size in x and y dimensions and the number of images)
    and a maximum of 256 square float32 images.
    For file names containing 'allDC', older versions of SimFCS 4 mistakenly
    write the header twice and report the wrong number of images.

    Parameters:
        filename:
            Name of file to open.
        dtype:
            Data type of image array.
        maxsize:
            Maximum square image length and number of images.
        doubleheader:
            File contains two copies of header.

    Examples:
        >>> with SimfcsZ64('simfcs.z64') as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs.z64.tif')
        ...     print(f.shape, data[142, 128, 128])
        ...
        (256, 256, 256) 2.0
        >>> with TiffFile('_simfcs.z64.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

        >>> with SimfcsZ64('simfcs_allDC.z64', doubleheader=True) as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs_allDC.z64.tif')
        ...     print(f.shape, data[128, 128])
        ...
        (256, 256) 172.0
        >>> with TiffFile('_simfcs_allDC.z64.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(z64|i64)$'
    _filesizemin = 32

    def _init(
        self,
        *,
        dtype: DTypeLike = '<f4',
        maxsize: tuple[int, int] = (256, 1024),
        doubleheader: bool = False,
        **kwargs: Any,
    ) -> None:
        """Read file header."""
        self._skip = 8 if doubleheader else 0
        header = self._decompress_header(self._skip + 8)
        header = header[self._skip : self._skip + 8]
        size, inum = struct.unpack('<ii', header)[:2]
        if not 2 <= size <= maxsize[-1] or not 2 <= inum <= maxsize[0]:
            raise LfdFileError(self, 'image size out of range')
        if inum == 1 or (doubleheader and 'allDC' in self._filename):
            self.shape = size, size
            self.axes = 'YX'
        else:
            self.shape = inum, size, size
            self.axes = 'QYX'
        self.dtype = numpy.dtype(dtype)

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return data as 3D array of float32."""
        assert (
            self._fh is not None
            and self.shape is not None
            and self.dtype is not None
        )
        bufsize = product(self.shape) * self.dtype.itemsize + 16
        rawdata = zlib.decompress(self._fh.read(), 15, bufsize)
        try:
            data = numpy.frombuffer(
                rawdata, '<' + self.dtype.char, offset=self._skip + 8
            )
            data = data.copy()  # make writable
            return data.reshape(*self.shape)
        except ValueError:
            return data[2:].reshape(*self.shape[1:])

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, compression='zlib', contiguous=False)
        data = self.asarray()
        # reshape according to axes provided by user
        axes = parse_kwargs(kwargs, axes=None)['axes']
        if axes:
            shape = {'ZYX': (0, None, 1, 2), 'TYX': (0, None, None, 1, 2)}[
                axes
            ]
            data.shape = tuple(
                1 if i is None else data.shape[i] for i in shape
            )
        tif.write(data, **kwargs)


def simfcsz64_write(
    filename: os.PathLike[Any] | str,
    data: ArrayLike,
    /,
) -> None:
    """Write stack of float32 images to Z64 file.

    Refer to :py:class:`SimfcsZ64` for the Z64 file format.

    Parameters:
        filename:
            Name of file to write.
        data:
            Data to write.
            Must be of shape (-1, size, size) and type float32.

    Examples:
        >>> data = (
        ...     numpy.arange(5 * 256 * 256).reshape(5, 256, 256).astype('f4')
        ... )
        >>> simfcsz64_write('_test.z64', data)
        >>> with SimfcsZ64('_test.z64') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """
    data = numpy.asarray(data)
    if data.dtype.char != 'f':
        raise ValueError(f'invalid data type {data.dtype} (must be float32)')
    if data.ndim != 3 or data.shape[1] != data.shape[2]:
        raise ValueError(f'invalid shape {data.shape}')
    rawdata = struct.pack('II', data.shape[2], data.shape[0]) + data.tobytes()
    rawdata = zlib.compress(rawdata)
    with open(filename, 'wb') as fh:
        fh.write(rawdata)


class SimfcsR64(SimfcsRef):
    """SimFCS compressed referenced fluorescence lifetime images.

    SimFCS R64 files contain referenced fluorescence lifetime images.
    The data are stored as a Zlib deflate compressed stream of one int32
    (defining the image size in x and y dimensions) and five (or more)
    square float32 images:

    0. dc - intensity
    1. ph1 - phase of 1st harmonic
    2. md1 - modulation of 1st harmonic
    3. ph2 - phase of 2nd harmonic
    4. md2 - modulation of 2nd harmonic

    Phase values are in degrees, the modulation values are normalized.
    Phase and modulation values may be `NaN`.

    Parameters:
        dtype:
            Data type of image array.
        maxsize:
            Maximum square length of image array.

    Examples:
        >>> with SimfcsR64('simfcs.r64') as f:
        ...     data = f.asarray()
        ...     f.totiff('_simfcs.r64.tif')
        ...     print(f.shape, data[:, 100, 200])
        ...
        (5, 256, 256) [0.25 23.22 0.642 104.3 2.117]
        >>> with TiffFile('_simfcs.r64.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.r64$'

    def _init(
        self,
        *,
        dtype: DTypeLike = '<f4',
        maxsize: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Read file header."""
        if self._filesize < 32:
            raise LfdFileError(self, 'file size out of range')
        size = struct.unpack('<i', self._decompress_header(4))[0]
        if not 2 <= size <= maxsize:
            raise LfdFileError(self, 'image size out of range')
        # can't determine real shape without decompressing whole file
        self.shape = 5, size, size
        self.dtype = numpy.dtype(dtype)
        self.axes = 'SYX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return data as 3D array of float32."""
        assert (
            self._fh is not None
            and self.shape is not None
            and self.dtype is not None
        )
        bufsize = product(self.shape) * self.dtype.itemsize + 4
        rawdata = zlib.decompress(self._fh.read(), bufsize=bufsize)
        data = numpy.frombuffer(rawdata, '<' + self.dtype.char, offset=4)
        data = data[: product(self.shape)].copy()  # make writable
        return data.reshape((-1, self.shape[1], self.shape[2]))

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(
            kwargs, compression='zlib', contiguous=False, metadata=False
        )
        label = 'dc'
        for i, image in enumerate(self.asarray()):
            if i > 0:
                i += 1
                label = f'ph{i // 2}' if i % 2 else f'md{i // 2}'
            tif.write(image, description=label, **kwargs)


def simfcsr64_write(
    filename: os.PathLike[Any] | str,
    data: ArrayLike,
    /,
) -> None:
    """Write referenced data to R64 file.

    Refer to :py:class:`SimfcsR64` for the format of referenced data.

    Parameters:
        filename:
            Name of file to write.
        data:
            Referenced data to write.
            Must be of shape (5, size, size) and type float32.

    Examples:
        >>> data = (
        ...     numpy.arange(5 * 256 * 256).reshape(5, 256, 256).astype('f4')
        ... )
        >>> simfcsr64_write('_test.r64', data)
        >>> with SimfcsR64('_test.r64') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """
    data = numpy.asarray(data)
    if data.dtype.char != 'f':
        raise ValueError(f'invalid data type {data.dtype} (must be float32)')
    if data.ndim != 3 or data.shape[0] < 5 or data.shape[1] != data.shape[2]:
        raise ValueError(f'invalid shape {data.shape}')
    rawdata = struct.pack('I', data.shape[-1]) + data.tobytes()
    rawdata = zlib.compress(rawdata)
    with open(filename, 'wb') as fh:
        fh.write(rawdata)


class SimfcsGpSeries(LfdFileSequence):
    """SimFCS generalized polarization image series.

    SimFCS GP series contain intensity images from two channels, stored in
    separate :py:class:`SimfcsInt` files with consecutive names.

    Examples:
        >>> ims = SimfcsGpSeries('gpint/v*.int')
        >>> ims.axes
        'CI'
        >>> ims = ims.asarray()
        >>> ims.shape
        (2, 135, 256, 256)

    """

    _readfunction = SimfcsInt
    _indexpattern = r'(?P<C>\d)(?P<I>\d+)\.int'


class FlimboxFbd(LfdFile):
    """FLIMbox data.

    FDB files contain encoded data from the FLIMbox device, storing a
    stream of 16-bit or 32-bit integers (data words) that can be decoded to
    photon arrival windows, channels, and times.
    FBD files are written by SimFCS and VistaVision.

    The measurement's frame size, pixel dwell time, number of sampling
    windows, and scanner type are encoded in the last four letters of the
    file name.
    Newer FBD files, where the 3rd character in the file name tag is `0`,
    start with the first 1kB of the firmware file used for the measurement,
    followed by 31kB containing a binary record with measurement settings.
    FBD files written by VistaVision are accompanied by FlimboxFbs setting
    files.

    It depends on the application and its setting how to interpret the
    decoded data, for example, as time series, line scans, or image frames
    of FCS or digital frequency domain fluorescence lifetime measurements.

    The data word format depends on the device's firmware.
    A common layout is::

        |F|E|D|C|B|A|9|8|7|6|5|4|3|2|1|0|  data word bits
                            |-----------|  pcc (cross correlation phase)
                        |---------------|  tcc (cross correlation time)
                      |-|                  marker (indicates start of frame)
        |-------------|                    index into decoder table

    The data word can be decoded into a cross correlation phase histogram
    index (shown for the 1st harmonics)::

        bin = (pmax-1 - (pcc + win * (pmax//windows)) % pmax) // pdiv

    - ``bin``, cross correlation phase index (phase histogram bin number).
    - ``pcc``, cross correlation phase (counter).
    - ``pmax``, number of entries in cross correlation phase histogram.
    - ``pdiv``, divisor to reduce number of entries in phase histogram.
    - ``win``, arrival window.
    - ``windows``, number of sampling windows.

    The current implementation uses a decoder table to decode arrival windows
    for each channel. This method is inefficient for 32-bit FLIMbox data
    with large number of windows and channels.

    Parameters:
        filename:
            Name of file to open.
        code:
            Four-character string, encoding frame size (1st char),
            pixel dwell time (2nd char), number of sampling windows
            (3rd char), and scanner type (4th char).
            By default, the code is extracted from the file name.
        frame_size:
            Number of pixels in one line scan, excluding retrace.
        windows:
            Number of sampling windows used by FLIMbox.
        channels:
            Number of channels used by FLIMbox.
        harmonics:
            First or second harmonics.
        pdiv:
            Divisor to reduce number of entries in phase histogram.
        pixel_dwell_time:
            Number of microseconds the scanner remains at each pixel.
        laser_frequency:
            Laser frequency in Hz.
            The default is 20000000 Hz, the internal FLIMbox frequency.
        laser_factor:
            Factor to correct dwell_time/laser_frequency.
            Use when the scanner clock is not known exactly.
        scanner_line_length:
            Number of pixels in each line, including retrace.
        scanner_line_start:
            Index of first valid pixel in scan line.
        scanner_frame_start:
            Index of first valid pixel after marker.
        scanner:
            Scanner software or hardware.
        synthesizer:
            Synthesizer software or hardware.
        **kwargs
            Additional arguments are ignored.

    Examples:
        >>> with FlimboxFbd('flimbox$CBCO.fbd') as f:
        ...     bins, times, markers = f.decode(
        ...         word_count=500000, skip_words=1900000
        ...     )
        ...
        >>> print(bins[0, :2], times[:2], markers)
        [53 51] [ 0 42] [ 44097 124815]
        >>> hist = [numpy.bincount(b[b >= 0]) for b in bins]
        >>> int(numpy.argmax(hist[0]))
        53

    """

    _filepattern = r'.*\.fbd$'
    _figureargs = {'figsize': (6, 5)}

    header: numpy.recarray[Any, Any] | None
    """FLIMbox file header, if any."""

    fbf: dict[str, Any] | None
    """FLIMbox firmware header settings, if any."""

    fbs: dict[str, Any] | None
    """FLIMbox settings from FBS.XML file, if any."""

    decoder: str | None
    """Decoder settings function."""

    code: str
    """Four-character string from file name, if any."""

    frame_size: int
    """Number of pixels in one line scan, excluding retrace."""

    windows: int
    """Number of sampling windows used by FLIMbox."""

    channels: int
    """Number of channels used by FLIMbox."""

    harmonics: int
    """First or second harmonics."""

    pdiv: int
    """Divisor to reduce number of entries in phase histogram."""

    pixel_dwell_time: float
    """Number of microseconds the scanner remains at each pixel."""

    laser_frequency: float
    """Laser frequency in Hz."""

    laser_factor: float
    """Factor to correct dwell_time/laser_frequency."""

    scanner_line_length: int
    """Number of pixels in each line, including retrace."""

    scanner_line_start: int
    """Index of first valid pixel in scan line."""

    scanner_frame_start: int
    """Index of first valid pixel after marker."""

    scanner: str
    """Scanner software or hardware."""

    synthesizer: str
    """Synthesizer software or hardware."""

    is_32bit: bool
    """Data words are 32-bit."""

    _data_offset: int

    _attributes = (
        'decoder',
        'frame_size',
        'windows',
        'channels',
        'harmonics',
        'pdiv',
        'pmax',
        'pixel_dwell_time',
        'laser_frequency',
        'laser_factor',
        'synthesizer',
        'scanner',
        'scanner_line_length',
        'scanner_line_start',
        'scanner_frame_start',
    )

    _frame_size: dict[str, int] = {
        # Map 1st character in file name tag to image frame size
        'A': 64,
        'B': 128,
        'C': 256,
        'D': 320,
        'E': 512,
        'F': 640,
        'G': 800,
        'H': 1024,
    }

    _flimbox_settings: dict[str, tuple[int, int, int]] = {
        # Map 3rd character in file name tag to (windows, channels, harmonics)
        # '0': file contains header
        'A': (2, 2, 1),
        'B': (4, 2, 1),
        'C': (8, 2, 1),
        'F': (8, 4, 1),
        'D': (16, 2, 1),
        'E': (32, 2, 1),
        'H': (64, 1, 1),
        # second harmonics. P and Q might be switched in some files?
        'N': (2, 2, 2),
        'O': (4, 2, 2),  # frequency = 40000000 ?
        'P': (8, 2, 2),
        'G': (8, 4, 2),
        'Q': (16, 2, 2),
        'R': (32, 2, 2),
        'S': (64, 1, 2),
    }

    _scanner_settings: dict[str, dict[str, Any]] = {
        # Map 4th and 2nd character in file name tag to pixel_dwell_time,
        # scanner_line_add, scanner_line_start, and scanner_frame_start.
        # As of SimFCS ~2011.
        # These values may not apply any longer and need to be overridden.
        'S': {
            'name': 'Native SimFCS, 3-axis card',
            'A': (4, 198, 99, 0),
            'J': (100, 20, 10, 0),
            'B': (5, 408, 204, 0),
            'K': (128, 16, 8, 0),
            'C': (8, 256, 128, 0),
            'L': (200, 10, 5, 0),
            'D': (10, 204, 102, 0),
            'M': (256, 8, 4, 0),
            'E': (16, 128, 64, 0),
            'N': (500, 4, 2, 0),
            'F': (20, 102, 51, 0),
            'O': (512, 4, 2, 0),
            'G': (32, 64, 32, 0),
            'P': (1000, 2, 1, 0),
            'H': (50, 40, 20, 0),
            'Q': (1024, 2, 1, 0),
            'I': (64, 32, 16, 0),
            'R': (2000, 1, 0, 0),
        },
        'O': {
            'name': 'Olympus FV 1000, NI USB',
            'A': (2, 10, 8, 0),
            'F': (20, 56, 55, 0),
            'B': (4, 10, 8, 0),
            'G': (40, 28, 20, 0),
            'C': (8, 10, 8, 0),
            'H': (50, 12, 10, 0),
            'D': (10, 112, 114, 0),
            'I': (100, 12, 9, 0),
            'E': (12.5, 90, 80, 0),
            'J': (200, 10, 89, 0),
        },
        'Y': {
            'name': 'Zeiss LSM510',
            'A': (6.39, 344, 22, 0),
            'D': (51.21, 176, 22, 414),
            'B': (12.79, 344, 22, 0),
            'E': (102.39, 88, 0, 242),
            'C': (25.61, 344, 22, 0),
            'F': (204.79, 12, 10, 0),
        },
        'Z': {
            'name': 'Zeiss LSM710',
            'A': (6.305, 344, 2, 0),
            'B': (12.79, 344, 2, 0),
            'C': (25.216, 344, 2, 0),
            'D': (50.42, 176, 8, 414),
            'E': (100.85, 88, 10, 242),
            'F': (177.32, 12, 10, 0),
        },
        'I': {
            'name': 'ISS Vista slow scanner',
            'A': (4, 112, 73, 0),
            'H': (32, 37, 28, 0),
            'B': (6, 112, 73, 0),
            'I': (40, 28, 19, 0),
            'C': (8, 112, 73, 0),
            'J': (64, 18, 14, 0),
            'D': (10, 112, 73, 0),
            'K': (200, 6, 4, 0),
            'E': (12.5, 90, 59, 0),
            'L': (500, 6, 4, 0),
            'F': (16, 73, 48, 0),
            'M': (1000, 6, 4, 0),
            'G': (20, 56, 37, 0),
        },
        'V': {
            'name': 'ISS Vista fast scanner',
            'A': (4, 112, 73, 0),
            'H': (32, 37, 28, 0),
            'B': (6, 112, 73, 0),
            'I': (40, 28, 19, 0),
            'C': (8, 112, 73, 0),
            'J': (64, 21, 14, 0),
            'D': (10, 112, 73, 0),
            'K': (100, 12, 8, 0),
            'E': (12.5, 90, 59, 0),
            'L': (200, 6, 4, 0),
            'F': (16, 73, 48, 0),
            'M': (500, 6, 4, 0),
            'G': (20, 56, 37, 0),
            'N': (1000, 6, 4, 0),
        },
        'T': {
            # used with new file format only?
            'name': 'IOTech scanner card'
        },
    }

    _header_t = [
        # Binary header starting at offset 1024 in files with $xx0x names.
        # This is written as a memory dump of a Delphi record, hence the pads
        ('owner', '<i4'),  # must be 0
        ('pixel_dwell_time_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('frame_size_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('line_length', '<i4'),
        ('points_end_of_frame', '<i4'),
        ('x_starting_pixel', '<i4'),
        ('line_integrate', '<i4'),
        ('scanner_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('synthesizer_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('windows_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('channels_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('_pad1', '<i4'),
        ('line_time', '<f8'),
        ('frame_time', '<f8'),
        ('scanner', '<i4'),
        ('_pad2', '<f4'),
        ('laser_frequency', '<f8'),
        ('laser_factor', '<f8'),
        ('frames_to_average', '<i4'),
        ('enabled_before_start', '<i4'),
        # the my?calib fields may be missing
        ('mypcalib', '<16f4'),
        ('mymcalib', '<16f4'),
        ('mypcalib1', '<16f4'),
        ('mymcalib1', '<16f4'),
        ('mypcalib2', '<16f4'),
        ('mymcalib2', '<16f4'),
        ('mypcalib3', '<16f4'),
        ('mymcalib3', '<16f4'),
        ('h1', '<i4'),
        ('h2', '<i4'),
        ('process_enable', 'b'),
        ('integrate', 'b'),
        ('detect_maitai', 'b'),
        ('trigger_on_up', 'b'),
        ('line_scan', 'b'),
        ('circular_scan', 'b'),
        ('average_frames_on_reading', 'b'),
        ('show_each_frame', 'b'),
        ('write_each_frame', 'b'),
        ('write_one_big_file', 'b'),
        ('subtract_background', 'b'),
        ('normalize_to_frames', 'b'),
        ('second_harmonic', 'b'),
        ('_pad3', '3b'),
        ('bin_pixel_by_index', '<i4'),
        ('jitter_level', '<i4'),
        ('phaseshift1', '<i4'),
        ('phaseshift2', '<i4'),
        ('acquire_item_index', '<i4'),
        ('acquire_number_of_frames', '<i4'),
        ('show_channel_index', '<i4'),
    ]

    _header_pixel_dwell_time = (
        # fmt: off
        1, 2, 4, 5, 8, 10, 16, 20, 32, 50, 64,
        100, 128, 200, 256, 500, 512, 1000, 2000,
        # fmt: on
    )

    _header_frame_size = (64, 128, 256, 320, 512, 640, 800, 1024)

    _header_bin_pixel_by = (1, 2, 4, 8)

    _header_windows = (4, 8, 16, 32, 64)

    _header_channels = (
        1,
        2,
        4,
        # the following are not supported
        4,  # 2 to 4 ch
        4,  # 2 ch 8w Spartan6
        4,  # 4 ch 16w frame
    )

    _header_scanners = (
        'IOTech scanner card',
        'Native SimFCS, 3-axis card',
        'Olympus FV 1000, NI USB',
        'Zeiss LSM510',
        'Zeiss LSM710',
        'ISS Vista slow scanner',
        'ISS Vista fast scanner',
        'M2 laptop only',
        'IOTech scanner card',
        'Leica SP8',
        'Zeiss LSM880',
        'Olympus FV3000',
        'Olympus 2P',
    )

    _header_synthesizers = (
        'Internal FLIMbox frequency',
        'Fianium',
        'Spectra Physics MaiTai',
        'Spectra Physics Tsunami',
        'Coherent Chameleon',
    )

    @cached_property
    def decoder_settings(self) -> dict[str, NDArray[numpy.int16] | int]:
        """Return parameters to decode FLIMbox data stream.

        Returns:
            - 'decoder_table' - Decoder table mapping channel and window
              indices to actual arrival windows.
              The shape is (channels, window indices) and dtype is int16.
            - 'tcc_mask', 'tcc_shr' - Binary mask and number of bits to right
              shift to extract cross correlation time from data word.
            - 'pcc_mask', 'pcc_shr' - Binary mask and number of bits to right
              shift to extract cross correlation phase from data word.
            - 'marker_mask', 'marker_shr' - Binary mask and number of bits to
              right shift to extract markers from data word.
            - 'win_mask', 'win_shr' - Binary mask and number of bits to right
              shift to extract index into lookup table from data word.

        """
        bytes_ = 4 if self.is_32bit else 2
        decoder = f'_b{bytes_}w{self.windows}c{self.channels}'
        if (
            self.windows == 16
            and self.channels == 4
            and self.fbf is not None
            and self.fbf.get('time', '').endswith('Bit')
        ):
            t = self.fbf['time'][:-3]
            decoder += f't{t}'
        self.decoder = decoder
        try:
            settings = getattr(self, decoder)()
        except Exception as exc:
            raise ValueError(f'Decoder{decoder} not implemented') from exc
        return settings  # type: ignore[no-any-return]

    @staticmethod
    def _b4w16c4t10() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 32-bit, 16 windows, 4 channels (Spartan6)
        # with line markers
        # 0b00000000000000000000000011111111 cross correlation phase
        # 0b00000000000000000000001111111111 cross correlation time  # 10 bit
        # 0b00000000000000000000010000000000 start of line marker
        # 0b00000000000000000000100000000000 start of frame marker
        # 0b00000000000000000001000000000000 ch0 photon
        # 0b00000000000000011110000000000000 ch0 window
        # 0b00000000000000100000000000000000 ch1 photon
        # 0b00000000001111000000000000000000 ch1 window
        # 0b00000000010000000000000000000000 ch2 photon
        # 0b00000111100000000000000000000000 ch2 window
        # 0b00001000000000000000000000000000 ch3 photon
        # 0b11110000000000000000000000000000 ch3 window
        table = numpy.full((4, 2**20), -1, numpy.int16)
        for i in range(2**20):
            if i & 0b1:
                table[0, i] = (i & 0b11110) >> 1
            if i & 0b100000:
                table[1, i] = (i & 0b1111000000) >> 6
            if i & 0b10000000000:
                table[2, i] = (i & 0b111100000000000) >> 11
            if i & 0b1000000000000000:
                table[3, i] = (i & 0b11110000000000000000) >> 16
        return {
            'decoder_table': table,
            'tcc_mask': 0x3FF,
            'tcc_shr': 0,
            'pcc_mask': 0xFF,
            'pcc_shr': 0,
            # 'line_mask': 0x400,
            # 'line_shr': 10,
            'marker_mask': 0x800,
            'marker_shr': 11,
            'win_mask': 0xFFFFF000,
            'win_shr': 12,
            'swap_words': True,
        }

    @staticmethod
    def _b4w16c4t11() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 32-bit, 16 windows, 4 channels (Spartan6)
        # without line markers
        # 0b00000000000000000000000011111111 cross correlation phase
        # 0b00000000000000000000011111111111 cross correlation time  # 11 bit
        # 0b00000000000000000000100000000000 start of frame marker
        # 0b00000000000000000001000000000000 ch0 photon
        # 0b00000000000000011110000000000000 ch0 window
        # 0b00000000000000100000000000000000 ch1 photon
        # 0b00000000001111000000000000000000 ch1 window
        # 0b00000000010000000000000000000000 ch2 photon
        # 0b00000111100000000000000000000000 ch2 window
        # 0b00001000000000000000000000000000 ch3 photon
        # 0b11110000000000000000000000000000 ch3 window
        return {
            **FlimboxFbd._b4w16c4t10(),
            'tcc_mask': 0x7FF,
            'marker_mask': 0x800,
            'marker_shr': 11,
        }

    @staticmethod
    def _b4w8c4() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 32-bit, 8 windows, 4 channels (Spartan-6)
        # 0b00000000000000000000000000000001 ch0 photon
        # 0b00000000000000000000000000001110 ch0 window
        # 0b00000000000000000000000000010000 ch1 photon
        # 0b00000000000000000000000011100000 ch1 window
        # 0b00000000000000000000000100000000 ch2 photon
        # 0b00000000000000000000111000000000 ch2 window
        # 0b00000000000000000001000000000000 ch3 photon
        # 0b00000000000000001110000000000000 ch3 window
        # 0b00000000111111110000000000000000 cross correlation phase
        # 0b00011111111111110000000000000000 cross correlation time
        # 0b01000000000000000000000000000000 start of frame marker
        table = numpy.full((4, 2**16), -1, numpy.int16)
        for i in range(2**16):
            if i & 0b1:
                table[0, i] = (i & 0b1110) >> 1
            if i & 0b10000:
                table[1, i] = (i & 0b11100000) >> 5
            if i & 0b100000000:
                table[2, i] = (i & 0b111000000000) >> 9
            if i & 0b1000000000000:
                table[3, i] = (i & 0b1110000000000000) >> 13
        return {
            'decoder_table': table,
            'tcc_mask': 0x1FFF0000,
            'tcc_shr': 16,
            'pcc_mask': 0xFF0000,
            'pcc_shr': 16,
            'marker_mask': 0x40000000,
            'marker_shr': 30,
            'win_mask': 0xFFFF,
            'win_shr': 0,
            # 'swap_words': True,  # ?
        }

    @staticmethod
    def _b2w4c2() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 4 windows, 2 channels
        # fmt: off
        table = numpy.array(
            [[-1, 0, -1, 1, -1, 2, -1, 3, -1, 0, -1, -1, 1, 0, -1, 2,
              1, 0, 3, 2, 1, 0, 3, 2, 1, -1, 3, 2, -1, -1, 3, -1],
             [-1, -1, 0, -1, 1, -1, 2, -1, 3, 0, -1, -1, 0, 3, -1, 0,
              1, 2, 2, 1, 2, 1, 1, 2, 3, -1, 0, 3, -1, -1, 3, -1]],
            numpy.int16
        )
        # fmt: on
        return {
            'decoder_table': table,
            'tcc_mask': 0x3FF,
            'tcc_shr': 0,
            'pcc_mask': 0xFF,
            'pcc_shr': 0,
            'marker_mask': 0x400,
            'marker_shr': 10,
            'win_mask': 0xF800,
            'win_shr': 11,
        }

    @staticmethod
    def _b2w8c2() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 8 windows, 2 channels
        # TODO: this does not correctly decode data acquired with ISS firmware
        table = numpy.full((2, 81), -1, numpy.int16)
        table[0, 1:9] = range(8)
        table[1, 9:17] = range(8)
        table[:, 17:] = numpy.mgrid[0:8, 0:8].reshape(2, -1)[::-1, :]
        return {
            'decoder_table': table,
            'tcc_mask': 0xFF,
            'tcc_shr': 0,
            'pcc_mask': 0x3F,
            'pcc_shr': 0,
            'marker_mask': 0x100,
            'marker_shr': 8,
            'win_mask': 0xFFFF,
            'win_shr': 9,
        }

    @staticmethod
    def _b2w8c4() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 8 windows, 4 channels
        logger().warning(
            'FlimboxFbd: b2w8c4 decoder not tested. '
            'Please submit a FBD file to https://github.com/cgohlke/lfdfiles'
        )
        table = numpy.full((4, 128), -1, numpy.int16)
        for i in range(128):
            win = i & 0b0000111
            ch0 = (i & 0b0001000) >> 3
            ch1 = (i & 0b0010000) >> 4
            ch2 = (i & 0b0100000) >> 5
            ch3 = (i & 0b1000000) >> 6
            if ch0 + ch1 + ch2 + ch3 != 1:
                continue
            if ch0:
                table[0, i] = win
            elif ch1:
                table[1, i] = win
            elif ch2:
                table[2, i] = win
            elif ch3:
                table[3, i] = win
        return {
            'decoder_table': table,
            'tcc_mask': 0xFF,
            'tcc_shr': 0,
            'pcc_mask': 0x3F,
            'pcc_shr': 0,
            'marker_mask': 0x100,
            'marker_shr': 8,
            'win_mask': 0xFFFF,
            'win_shr': 9,
        }

    @staticmethod
    def _b2w16c1() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 16 windows, 1 channel
        logger().warning(
            'FlimboxFbd: b2w16c1 decoder not tested. '
            'Please submit a FBD file to https://github.com/cgohlke/lfdfiles'
        )
        table = numpy.full((1, 32), -1, numpy.int16)
        for i in range(32):
            win = (i & 0b11110) >> 1
            ch0 = i & 0b00001
            if ch0:
                table[0, i] = win
        return {
            'decoder_table': table,
            'tcc_mask': 0xFF,
            'tcc_shr': 0,
            'pcc_mask': 0x3F,
            'pcc_shr': 0,
            'marker_mask': 0x100,
            'marker_shr': 8,
            'win_mask': 0xFFFF,
            'win_shr': 11,
        }

    @staticmethod
    def _b2w16c2() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 16 windows, 2 channels
        logger().warning(
            'FlimboxFbd: b2w16c2 decoder not tested. '
            'Please submit a FBD file to https://github.com/cgohlke/lfdfiles'
        )
        table = numpy.full((2, 64), -1, numpy.int16)
        for i in range(64):
            win = (i & 0b111100) >> 2
            ch0 = (i & 0b000010) >> 1
            ch1 = i & 0b000001
            if ch0 + ch1 != 1:
                continue
            if ch0:
                table[0, i] = win
            elif ch1:
                table[1, i] = win
        return {
            'decoder_table': table,
            'tcc_mask': 0xFF,
            'tcc_shr': 0,
            'pcc_mask': 0x3F,
            'pcc_shr': 0,
            'marker_mask': 0x100,
            'marker_shr': 8,
            'win_mask': 0xFFFF,
            'win_shr': 10,
        }

    @staticmethod
    def _b2w32c2() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 32 windows, 2 channels
        # TODO
        raise NotImplementedError(
            'FlimboxFbd: b2w32c2 decoder not implemented. '
            'Please submit a FBD file to https://github.com/cgohlke/lfdfiles'
        )

    @staticmethod
    def _b2w64c1() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 64 windows, 1 channel
        # TODO
        raise NotImplementedError(
            'FlimboxFbd: b2w64c1 decoder not implemented. '
            'Please submit a FBD file to https://github.com/cgohlke/lfdfiles'
        )

    def _init(
        self,
        code: str = '',
        frame_size: int = -1,
        windows: int = -1,
        channels: int = -1,
        harmonics: int = -1,
        pdiv: int = -1,
        pixel_dwell_time: float = -1.0,
        laser_frequency: float = -1.0,
        laser_factor: float = -1.0,
        scanner_line_length: int = -1,
        scanner_line_start: int = -1,
        scanner_frame_start: int = -1,
        scanner: str = '',
        synthesizer: str = '',
        **kwargs: Any,
    ) -> None:
        """Initialize instance from file name code or file header."""
        self.decoder = None
        self.fbf = None
        self.fbs = None
        self.header = None
        self.code = code
        self.frame_size = frame_size
        self.windows = windows
        self.channels = channels
        self.harmonics = harmonics
        self.pdiv = pdiv
        self.pixel_dwell_time = pixel_dwell_time
        self.laser_frequency = laser_frequency
        self.laser_factor = laser_factor
        self.scanner_line_length = scanner_line_length
        self.scanner_line_start = scanner_line_start
        self.scanner_frame_start = scanner_frame_start
        self.scanner = scanner
        self.synthesizer = synthesizer
        self.is_32bit = False
        self._data_offset = 0

        if not self.code:
            match = re.search(
                r'.*\$([A-Z][A-Z][A-Z0-9][A-Z])\.fbd',
                self._filename,
                re.IGNORECASE,
            )
            if match is None:
                self.code = 'CFCS'  # old FLIMbox file ?
                warnings.warn(
                    'FlimboxFbd: failed to parse code from file name. '
                    f'Using {self.code!r}'
                )
            else:
                self.code = match.group(1)

        assert len(self.code) == 4, code

        if self.code[2].isnumeric() and self._from_fbs():
            pass
        elif self.code[2] == '0':
            self._from_header()
        else:
            self._from_code()

        assert self.windows >= 0
        assert self.channels >= 0
        assert self.harmonics >= 0

        if self.pdiv <= 0:
            try:
                self.pdiv = max(1, self.pmax // 64)
            except AttributeError:
                pass
        assert self.pdiv >= 0

        for attr in self._attributes:
            value = getattr(self, attr)
            if isinstance(value, (int, float)):
                if value < 0:
                    raise ValueError(f"'{attr}' not initialized")
            elif not value:
                # empty str
                raise ValueError(f"'{attr}' not initialized")

    def _from_fbs(self) -> bool:
        """Initialize instance from VistaVision settings file."""
        fname = self.filename.rsplit('$', 1)[0] + '.fbs.xml'
        if not os.path.exists(fname):
            return False
        # self.pdiv = 1
        with FlimboxFbs(fname) as fbs:
            scn = fbs['ScanParams']
            self.fbf = fbf = _flimboxfbf_parse(
                fbs['FirmwareParams']['Description']
            )
            self.is_32bit = '32fifo' in fbf['decoder']
            if self.harmonics < 0:
                self.harmonics = (1, 2)[fbf['secondharmonic']]
            if self.windows < 0:
                self.windows = fbf['windows']
            if self.channels < 0:
                self.channels = fbf['channels']
            if self.synthesizer == '':
                self.synthesizer = 'Unknown'
            if self.scanner == '':
                try:
                    self.scanner = scn['ScannerInfo']['ScannerID']
                except IndexError:
                    self.scanner = 'Unknown'
            if self.laser_frequency < 0:
                self.laser_frequency = float(scn['ExcitationFrequency'])
            if self.laser_factor < 0:
                self.laser_factor = 1.0
            if self.scanner_line_length < 0:
                self.scanner_line_length = int(scn['ScanLineLength'])
            if self.scanner_line_start < 0:
                self.scanner_line_start = int(scn['ScanLineLeftBorder'])
            if 'PixelOffsetToFrameFlag' in scn:
                # TODO: check this
                self.scanner_frame_start = scn['PixelOffsetToFrameFlag']
            else:
                self.scanner_frame_start = 0
            if self.pixel_dwell_time < 0:
                self.pixel_dwell_time = scn['PixelDwellTime']['value']
                if (
                    'Unit' in scn['PixelDwellTime']
                    and 'milli' in scn['PixelDwellTime']['Unit'].lower()
                ):
                    self.pixel_dwell_time *= 1000
            if self.frame_size < 0:
                self.frame_size = scn['XPixels']
            self.fbs = fbs.asdict()
        self._fh.seek(0)
        try:
            if self._fh.read(32).decode('cp1252').isprintable():
                # header detected, assume encoded data starts at 33K
                self._data_offset = 33792
        except UnicodeDecodeError:
            pass
        self._fh.seek(0)
        return True

    def _from_code(self) -> None:
        """Initialize instance from file name code."""
        code = self.code
        if self.frame_size < 0:
            self.frame_size = self._frame_size[code[0]]
        if self.windows < 0 or self.channels < 0 or self.harmonics < 0:
            windows, channels, harmonics = self._flimbox_settings[code[2]]
            if self.windows < 0:
                self.windows = windows
            if self.channels < 0:
                self.channels = channels
            if self.harmonics < 0:
                self.harmonics = harmonics
        if (
            self.pixel_dwell_time < 0
            or self.scanner_line_length < 0
            or self.scanner_line_start < 0
            or self.scanner_frame_start < 0
        ):
            (
                pixel_dwell_time,
                scanner_line_add,
                scanner_line_start,
                scanner_frame_start,
            ) = self._scanner_settings[code[3]][code[1]]
            if self.pixel_dwell_time < 0:
                self.pixel_dwell_time = pixel_dwell_time
            if self.scanner_frame_start < 0:
                self.scanner_frame_start = scanner_frame_start
            if self.scanner_line_start < 0:
                self.scanner_line_start = scanner_line_start
            if self.scanner_line_length < 0:
                self.scanner_line_length = self.frame_size + scanner_line_add
        if self.scanner == '':
            self.scanner = self._scanner_settings[code[3]]['name']
        if self.synthesizer == '':
            self.synthesizer = 'Unknown'
        if self.laser_frequency < 0:
            self.laser_frequency = 20000000 * self.harmonics
        if self.laser_factor < 0:
            self.laser_factor = 1.0

    def _from_header(self) -> None:
        """Initialize instance from 32 KB file header."""
        # the first 1024 bytes contain the start of a FLIMbox firmware file
        with FlimboxFbf(self._fh.name, validate=False) as fbf:
            self.fbf = fbf.asdict()
        assert self.fbf is not None
        assert self._fh is not None

        self.is_32bit = '32fifo' in self.fbf['decoder']

        # the next 31kB contain the binary file header
        self._fh.seek(1024)
        self.header = hdr = numpy.fromfile(self._fh, self._header_t, 1)[0]
        if hdr['owner'] != 0:
            raise ValueError(f"unknown header format '{hdr['owner']}'")

        # detect corrupted header: the my?calib fields may be missing
        if (
            hdr['process_enable'] > 1
            or hdr['integrate'] > 1
            or hdr['line_scan'] > 1
            or hdr['subtract_background'] > 1
            or hdr['second_harmonic'] > 1
        ):
            # reload modified header
            self._header_t = (
                FlimboxFbd._header_t[:20] + FlimboxFbd._header_t[28:]
            )
            self._fh.seek(1024)
            self.header = hdr = numpy.fromfile(self._fh, self._header_t, 1)[0]

        if self.harmonics < 0:
            self.harmonics = (1, 2)[self.fbf['secondharmonic']]
        if self.windows < 0:
            windows = self._header_windows[hdr['windows_index']]
            self.windows = self.fbf.get('windows', windows)
            if self.windows != windows:
                logger().warning(
                    'FlimboxFbd: '
                    'windows mismatch between FBF and FBD header '
                    f'({self.windows!r} != {windows!r})'
                )
        if self.channels < 0:
            channels = self._header_channels[hdr['channels_index']]
            self.channels = self.fbf.get('channels', channels)
            if self.channels != channels:
                logger().warning(
                    'FlimboxFbd: '
                    'channels mismatch between FBF and FBD header '
                    f'({self.channels!r} != {channels!r})'
                )
        if self.synthesizer == '':
            try:
                self.synthesizer = self._header_synthesizers[
                    hdr['synthesizer_index']
                ]
            except IndexError:
                self.synthesizer = 'Unknown'
        if self.scanner == '':
            try:
                self.scanner = self._header_scanners[hdr['scanner_index']]
            except IndexError:
                self.scanner = 'Unknown'
        if self.laser_frequency < 0:
            self.laser_frequency = float(hdr['laser_frequency'])
        if self.laser_factor < 0:
            self.laser_factor = float(hdr['laser_factor'])
        if self.scanner_line_length < 0:
            self.scanner_line_length = int(hdr['line_length'])
        if self.scanner_line_start < 0:
            self.scanner_line_start = int(hdr['x_starting_pixel'])
        self.scanner_frame_start = max(self.scanner_frame_start, 0)

        if hdr['frame_time'] >= hdr['line_time'] > 1.0:
            if self.frame_size < 0:
                self.frame_size = round(hdr['frame_time'] / hdr['line_time'])
                for frame_size in self._header_frame_size:
                    if abs(self.frame_size - frame_size) < 3:
                        self.frame_size = frame_size
                        break
            if self.pixel_dwell_time < 0:
                self.pixel_dwell_time = float(
                    hdr['frame_time']
                    / (self.frame_size * self.scanner_line_length)
                )
        else:
            if self.frame_size < 0:
                try:
                    self.frame_size = round(
                        self._header_frame_size[hdr['frame_size_index']]
                    )
                except IndexError:
                    # fall back to filename
                    if self.code:
                        self.frame_size = self._frame_size[self.code[0]]
            if self.pixel_dwell_time < 0:
                try:
                    # use file name
                    dwt = self._scanner_settings[self.code[3]][self.code[1]][0]
                except (IndexError, KeyError, TypeError):
                    try:
                        dwt = self._header_pixel_dwell_time[
                            hdr['pixel_dwell_time_index']
                        ]
                    except IndexError:
                        dwt = 1.0
                self.pixel_dwell_time = dwt

        self._data_offset = 65536  # start of encoded data; contains 2 headers

    @property
    def pmax(self) -> int:
        """Number of entries in cross correlation phase histogram."""
        d = self.decoder_settings
        return int((d['pcc_mask'] >> d['pcc_shr']) + 1) // self.harmonics

    @property
    def scanner_line_add(self) -> int:
        """Number of pixels added to each line (for retrace)."""
        return self.scanner_line_length - self.frame_size

    @property
    def units_per_sample(self) -> float:
        """Number of FLIMbox units per scanner sample."""
        return float(
            (self.pixel_dwell_time * 1e-6)
            * (self.pmax / (self.pmax - 1))
            * (self.laser_frequency * self.laser_factor)
        )

    def decode(
        self,
        data: NDArray[Any] | None = None,
        *,
        word_count: int = -1,
        skip_words: int = 0,
        max_markers: int = 65536,
        **kwargs: Any,
    ) -> tuple[
        NDArray[numpy.int8 | numpy.int16],
        NDArray[numpy.uint32 | numpy.uint64],
        NDArray[numpy.intp],
    ]:
        """Return decoded FLIMbox data stream.

        Parameters:
            data:
                FLIMbox data stream. By default, the data is read from file.
            word_count:
                Number of data words to process.
                By default, all words are processed.
            skip_words:
                Number of data words to skip at beginning of stream.
            max_markers:
                Maximum number of markers expected in data stream.

        Returns:
            bins:
                Cross correlation phase index for all channels and data points.
                A int8 array of shape (channels, size).
                A value of -1 means no photon was counted.
            times:
                The times in FLIMbox counter units at each data point.
                An array of type uint64 or uint32, potentially huge.
            markers:
                The indices of up markers in the data stream, usually
                indicating frame starts. An array of type ssize_t.

        """
        assert self._fh is not None
        dtype = numpy.dtype('<u4' if self.is_32bit else '<u2')
        if data is None:
            self._fh.seek(self._data_offset + skip_words * dtype.itemsize, 0)
            data = numpy.fromfile(self._fh, dtype=dtype, count=word_count)
        elif skip_words or word_count != -1:
            if word_count < 0:
                data = data[skip_words:word_count]
            else:
                data = data[skip_words : skip_words + word_count]
        if data.dtype != dtype:
            raise ValueError('invalid data dtype')

        bins_out = numpy.empty(
            (self.channels, data.size),
            dtype=numpy.int8 if self.pmax // self.pdiv <= 128 else numpy.int16,
        )
        times_out = numpy.empty(data.size, dtype=numpy.uint64)
        markers_out = numpy.zeros(max_markers, dtype=numpy.intp)

        flimbox_decode(
            data,
            bins_out,
            times_out,
            markers_out,
            self.windows,
            self.pdiv,
            self.harmonics,
            **self.decoder_settings,
        )

        markers_out = markers_out[markers_out > 0]  # type: ignore[assignment]
        if len(markers_out) == max_markers:
            warnings.warn(
                f'number of markers exceeded buffer size {max_markers}'
            )

        return bins_out, times_out, markers_out

    def frames(
        self,
        decoded: tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | None,
        /,
        *,
        select_frames: slice | None = None,
        aspect_range: tuple[float, float] = (0.8, 1.2),
        frame_cluster: int = 0,
        **kwargs: Any,
    ) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        """Return shape and start/stop indices of scanner frames.

        If unable to detect any frames using the default settings, try to
        determine a correction factor from clusters of frame durations.

        Parameters:
            decoded:
                Times and markers as returned by the decode function.
                If None, call :py:meth:`FlimboxFbd.decode`.
            select_frames:
                Specifies which image frames to return.
                By default, all frames are returned.
            aspect_range:
                Minimum and maximum aspect ratios of valid frames.
                The default lets 1:1 aspect pass.
            frame_cluster:
                Index of the frame duration cluster to use when calculating
                the correction factor.
            **kwargs:
                Additional arguments passed to :py:meth:`FlimboxFbd.decode`.

        Returns:
            1. shape: Dimensions of scanner frame.
            2. frame_markers: Start and stop indices of detected image frames.

        """
        if decoded is None:
            decoded = self.decode(**kwargs)
        times, markers = decoded[-2:]

        line_time = self.scanner_line_length * self.units_per_sample
        frame_durations = numpy.ediff1d(times[markers])

        frame_markers = []
        if aspect_range:
            # detect frame markers assuming correct settings
            line_num = sys.maxsize
            for i, duration in enumerate(frame_durations):
                lines = duration / line_time
                aspect = self.frame_size / lines
                if aspect_range[0] < aspect < aspect_range[1]:
                    frame_markers.append(
                        (int(markers[i]), int(markers[i + 1]) - 1)
                    )
                else:
                    continue
                line_num = min(line_num, lines)
            line_num = int(round(line_num))

        if not frame_markers:
            # calculate frame duration clusters, assuming few clusters that
            # are narrower and more separated than cluster_size.
            cluster_size = 1024
            clusters: list[list[int]] = []
            cluster_indices = []
            for d in frame_durations:
                d = int(d)
                for i, c in enumerate(clusters):
                    if abs(d - c[0]) < cluster_size:
                        cluster_indices.append(i)
                        c[0] = min(c[0], d)
                        c[1] += 1
                        break
                else:
                    cluster_indices.append(len(clusters))
                    clusters.append([d, 1])
            clusters = list(sorted(clusters, key=lambda x: x[1], reverse=True))
            # possible correction factors, assuming square frame shape
            line_num = self.frame_size
            laser_factors = [c[0] / (line_time * line_num) for c in clusters]
            # select specified frame cluster
            frame_cluster = min(frame_cluster, len(laser_factors) - 1)
            self.laser_factor = laser_factors[frame_cluster]
            frame_markers = [
                (int(markers[i]), int(markers[i + 1]) - 1)
                for i, c in enumerate(cluster_indices)
                if c == frame_cluster
            ]
            msg = [
                'no frames detected with default settings.'
                'Using square shape and correction factor '
                f'{self.laser_factor:.5f}.'
            ]
            if len(laser_factors) > 1:
                factors = ', '.join(f'{i:.5f}' for i in laser_factors[:4])
                msg.append(
                    f'The most probable correction factors are: {factors}'
                )
            warnings.warn('\n'.join(msg))

        if not isinstance(select_frames, slice):
            select_frames = slice(select_frames)
        frame_markers = frame_markers[select_frames]
        if not frame_markers:
            raise ValueError('no frames selected')
        return (line_num, self.scanner_line_length), frame_markers

    def asimage(
        self,
        decoded: tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | None,
        frames: tuple[tuple[int, int], list[tuple[int, int]]] | None,
        /,
        *,
        integrate_frames: int = 1,
        square_frame: bool = True,
        **kwargs: Any,
    ) -> NDArray[numpy.uint16]:
        """Return image histograms from decoded data and detected frames.

        This function may fail to produce expected results when settings
        were recorded incorrectly, scanner and FLIMbox frequencies were out
        of sync, or scanner settings were changed during acquisition.

        Parameters:
            decoded:
                Bins, times, and markers as returned by the decode function.
                If None, call :py:meth:`FlimboxFbd.decode`.
            frames:
                Scanner_shape and frame_markers as returned by the frames
                function.
                If None, call :py:meth:`FlimboxFbd.frames`.
            integrate_frames:
                Specifies which frames to sum. By default, all frames are
                summed into one. If 0, no frames are summed.
            square_frame:
                If True, return square image (frame_size x frame_size),
                else return full scanner frame.
            **kwargs:
                Additional arguments passed to :py:meth:`FlimboxFbd.decode`.

        Returns:
            Image histogram of shape (number of frames, channels in bins
            array, detected line numbers, frame_size, histogram bins).

        """
        if decoded is None:
            decoded = self.decode(**kwargs)
        bins, times, markers = decoded
        if frames is None:
            frames = self.frames(decoded, **kwargs)
        scanner_shape, frame_markers = frames
        # an extra line to scanner frame to allow clipping indices
        scanner_shape = scanner_shape[0] + 1, scanner_shape[1]
        # allocate output array of scanner frame shape
        shape = (
            integrate_frames if integrate_frames else len(frame_markers),
            bins.shape[0],  # channels
            scanner_shape[0] * scanner_shape[1],
            self.pmax // self.pdiv,
        )
        result = numpy.zeros(shape, dtype=numpy.uint16)
        # calculate frame data histogram
        flimbox_histogram(
            bins,
            times,
            frame_markers,
            self.units_per_sample,
            self.scanner_frame_start,
            result,
        )
        # reshape frames and slice valid region
        result = result.reshape(shape[:2] + scanner_shape + shape[-1:])
        if square_frame:
            result = result[
                ...,
                : self.frame_size,
                self.scanner_line_start : self.scanner_line_start
                + self.frame_size,
                :,
            ]
        return result

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.int8 | numpy.int16]:
        """Return cross correlation phase index of shape (channels, size).

        A value of -1 means no photon was counted.

        """
        return self.decode(data=None, **kwargs)[0]

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Plot lifetime histogram for all channels."""
        assert pyplot is not None
        assert self.pmax is not None
        assert self.pdiv is not None
        ax = figure.add_subplot(1, 1, 1)
        ax.set_title(self._filename)
        ax.set_xlabel('Bin')
        ax.set_ylabel('Counts')
        ax.set_xlim((0, self.pmax // self.pdiv - 1))
        bins, times, markers = self.decode()
        bins_channel: Any  # for mypy
        for ch, bins_channel in enumerate(bins):
            histogram = numpy.bincount(bins_channel[bins_channel >= 0])
            ax.plot(histogram, label=f'Ch{ch}')
        ax.legend()

    def _str(self) -> str | None:
        """Return additional information about file."""
        info = [f'{a}: {getattr(self, a)}' for a in self._attributes]
        if self.fbf is not None:
            info.append(
                indent(
                    'firmware:',
                    *(f'{k}: {v}'[:64] for k, v in self.fbf.items()),
                )
            )
        if self.header is not None:
            info.append(
                indent(
                    'header:',
                    *(
                        f'{k}: {self.header[k]}'[:64]
                        for k, _ in self._header_t
                        if k[0] != '_'
                    ),
                )
            )
        info.append(
            indent(
                'decoder_settings:',
                *(
                    (
                        f'{k}: {v:#x}'
                        if 'mask' in k
                        else f'{k}: {v}'[:64].splitlines()[0]
                    )
                    for k, v in self.decoder_settings.items()
                ),
            )
        )
        return '\n'.join(info)


class FlimboxFbs(LfdFile):
    """FLIMbox settings.

    VistaVision FBS.XML files contain FLIMbox acquisition settings in XML
    format.

    The properties can be accessed via dictionary interface.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with FlimboxFbs('flimbox.fbs.xml') as f:
        ...     f['ScanParams']['ExcitationFrequency']
        ...
        20000000

    """

    _filemode = 'r'
    _fileencoding = 'utf-8'
    _filepattern = r'.*\.(fbs.xml)$'
    _noplot = True

    _settings: dict[str, Any]

    def _init(self, **kwargs: Any) -> None:
        """Read and parse XML."""
        assert self._fh is not None
        buf = self._fh.read(100)
        if not ('<?xml ' in buf and '<FastFlimFbdDataSettings>' in buf):
            raise LfdFileError(self)
        self._fh.seek(0)
        buf = self._fh.read()
        info = tifffile.xml2dict(buf)
        self._settings = info['FastFlimFbdDataSettings']
        if not self._settings:
            raise LfdFileError(self)

    def asdict(self) -> dict[str, Any]:
        """Return settings as dict."""
        return copy.deepcopy(self._settings)

    def _asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Raise ValueError."""
        raise ValueError('FlimboxFbs file does not contain array data')

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        raise ValueError('FlimboxFbs file does not contain image data')

    def _str(self) -> str | None:
        """Return string with settings."""
        return format_dict(self._settings)

    def __getitem__(self, key: str, /) -> Any:
        return self._settings[key]

    def __contains__(self, key: str, /) -> bool:
        return key in self._settings

    def __len__(self) -> int:
        return len(self._settings)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._settings)


class FlimboxFbf(LfdFile):
    """FLIMbox firmware.

    FBF files contain FLIMbox device firmwares, stored in binary form
    following a NULL terminated ASCII string containing properties and
    description.

    The properties (lower cased) can be accessed via dictionary interface.

    Parameters:
        filename:
            Name of file to open.
        maxheaderlength:
            Maximum length of header.

    Examples:
        >>> with FlimboxFbf('flimbox.fbf') as f:
        ...     f['windows']
        ...     f['channels']
        ...     f['secondharmonic']
        ...     'extclk' in f
        ...
        16
        2
        0
        True

    """

    _filepattern = r'.*\.(fbf)$'
    _noplot = True

    _settings: dict[str, Any]

    def _init(self, *, maxheaderlength: int = 1024, **kwargs: Any) -> None:
        """Read and parse NULL terminated header string."""
        assert self._fh is not None
        try:
            # the first 1024 bytes contain the header
            header = self._fh.read(maxheaderlength).split(b'\x00', 1)[0]
        except ValueError as exc:
            raise LfdFileError(self) from exc
        header = bytes2str(header)
        if len(header) != maxheaderlength:
            self._fh.seek(len(header) + 1)
        self._settings = _flimboxfbf_parse(header)
        if not self._settings:
            raise LfdFileError(self)

    def firmware(self) -> bytes:
        """Return firmware as binary string."""
        assert self._fh is not None
        return self._fh.read()  # type: ignore[no-any-return]

    def asdict(self) -> dict[str, Any]:
        """Return settings as dict."""
        return copy.deepcopy(self._settings)

    def _asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Raise ValueError."""
        raise ValueError('FlimboxFbf file does not contain array data')

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        raise ValueError('FlimboxFbf file does not contain image data')

    def _str(self) -> str | None:
        """Return string with header settings."""
        return format_dict(self._settings)

    def __getitem__(self, key: str, /) -> Any:
        return self._settings[key]

    def __contains__(self, key: str, /) -> bool:
        return key in self._settings

    def __len__(self) -> int:
        return len(self._settings)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._settings)


def _flimboxfbf_parse(header: str, /) -> dict[str, Any]:
    """Return FLIMbox firmware settings from header."""
    settings: dict[str, Any] = {}
    try:
        header, comment_ = header.rsplit('/', 1)
        comment = [comment_]
    except ValueError:
        comment = []
    for name, value, unit in re.findall(
        r'([a-zA-Z\s]*)([.\d]*)([a-zA-Z\d]*)/', header
    ):
        name = name.strip().lower()
        if not name:
            name = {'w': 'windows', 'ch': 'channels'}.get(unit, None)
            unit = ''
        if not name:
            comment.append(name + value + unit)
            continue
        if unit == 'MHz':
            unit = 1000000
        try:
            if unit:
                value = int(value) * int(unit)
            else:
                value = int(value)
            unit = 0
        except ValueError:
            pass
        settings[name] = (value + unit) if value != '' else True
    cstr = '/'.join(reversed(comment))
    if cstr:
        settings['comment'] = cstr
    return settings


def _flimbox_decode(
    data: Any,
    bins_out: Any,
    times_out: Any,
    markers_out: Any,
    windows: Any,
    pdiv: Any,
    harmonics: Any,
    decoder_table: Any,
    tcc_mask: Any,
    tcc_shr: Any,
    pcc_mask: Any,
    pcc_shr: Any,
    marker_mask: Any,
    marker_shr: Any,
    win_mask: Any,
    win_shr: Any,
) -> None:
    """Decode FLIMbox data stream.

    See the documentation of the FlimboxFbd class for parameter descriptions
    and the lfdfiles.pyx file for a faster implementation.

    """
    # this implementation is for reference only. Do not use
    from ._lfdfiles import flimbox_decode  # noqa

    tcc = data & tcc_mask  # cross correlation time
    if tcc_shr:
        tcc >>= tcc_shr
    times_out[:] = tcc
    times_out[times_out == 0] = (tcc_mask >> tcc_shr) + 1
    times_out[0] = 0
    times_out[1:] -= tcc[:-1]
    del tcc
    numpy.cumsum(times_out, out=times_out)

    markers = data & marker_mask
    markers = numpy.diff(markers.view(numpy.int16))
    markers = numpy.where(markers > 0)[0]  # trigger up
    markers += 1
    size = min(len(markers), len(markers_out))
    markers_out[:size] = markers[:size]
    del markers

    if win_mask != 0xFFFF:  # window index
        win = data & win_mask
        win >>= win_shr
    else:
        win = data >> win_shr
    win = decoder_table.take(win, axis=1)
    nophoton = win == -1
    pmax = (pcc_mask >> pcc_shr + 1) // harmonics
    win *= pmax // windows * harmonics
    pcc = data & pcc_mask  # cross correlation phase
    if pcc_shr:
        pcc >>= pcc_shr
    win += pcc
    del pcc
    win %= pmax
    win += 1 - pmax
    numpy.negative(win, win)
    if pdiv > 1:
        win //= pdiv
    win = win.astype(numpy.int8)
    win[nophoton] = -1
    bins_out[:] = win


def _flimbox_histogram(
    bins: Any,
    times: Any,
    frame_markers: Any,
    units_per_sample: Any,
    scanner_frame_start: Any,
    out: Any,
) -> None:
    """Calculate histograms from decoded FLIMbox data and frame markers.

    See the documentation of the FlimboxFbd class for parameter descriptions
    and the _lfdfiles.pyx file for a much faster implementation.

    """
    # this implementation is for reference only. Do not use
    from ._lfdfiles import flimbox_histogram  # noqa

    nframes, nchannels, frame_length, nwindows = out.shape
    for f, (j, k) in enumerate(frame_markers):
        f = f % nframes
        t = times[j:k] - times[j]
        t /= units_per_sample  # index into flattened array
        if scanner_frame_start:
            t -= scanner_frame_start
        t = t.astype(numpy.uint32, copy=False)
        numpy.clip(t, 0, frame_length - 1, out=t)
        for c in range(nchannels):
            d = bins[c, j:k]
            for w in range(nwindows):
                x = numpy.where(d == w)[0]
                x = t.take(x)
                x = numpy.bincount(x, minlength=frame_length)
                out[f, c, :, w] += x


try:
    from ._lfdfiles import flimbox_decode, flimbox_histogram, sflim_decode
except ImportError:

    def flimbox_histogram(*args: Any, **kwargs: Any) -> None:
        """Raise ImportError: No module named '_lfdfiles'."""
        from ._lfdfiles import flimbox_histogram  # noqa

    def flimbox_decode(*args: Any, **kwargs: Any) -> None:
        """Raise ImportError: No module named '_lfdfiles'."""
        from ._lfdfiles import flimbox_decode  # noqa

    def sflim_decode(*args: Any, **kwargs: Any) -> None:
        """Raise ``ImportError: No module named '_sflim'``."""
        from ._lfdfiles import sflim_decode  # noqa


# deprecated alisases
SimfcsFbd = FlimboxFbd
SimfcsFbf = FlimboxFbf
simfcsfbd_histogram = flimbox_histogram
simfcsfbd_decode = flimbox_decode


class GlobalsLif(LfdFile):
    """Globals binary lifetime data.

    Globals LIF files contain array and meta data of multiple frequency-domain
    cuvette lifetime measurement, stored as consecutive 472-byte records.
    The number of frequencies per record is limited to 25. The format was
    also used by ISS software.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with GlobalsLif('globals.lif') as f:
        ...     print(len(f), f[42]['date'], f[42].asarray().shape)
        ...
        43 1987.8.8 (5, 11)

    """

    _filepattern = r'.*\.lif$'

    _records: list[Record]
    _record_t = numpy.dtype(
        [
            ('_title_len', 'u1'),
            ('title', 'S80'),
            ('number', 'i2'),
            ('frequency', [('_len', 'u1'), ('str', 'S6')], 25),
            ('phase', 'i2', 25),
            ('modulation', 'i2', 25),
            ('deltap', 'i2', 25),
            ('deltam', 'i2', 25),
            ('nanal', 'i2'),
            ('date', 'i2', 3),
            ('time', 'i2', 3),
        ]
    )

    class Record(dict[str, object]):
        """Record in GlobalsLif files."""

        def asarray(self) -> NDArray[Any]:
            """Return record array data."""
            return numpy.array(
                (
                    self['frequency'],
                    self['phase'],
                    self['modulation'],
                    self['deltap'],
                    self['deltam'],
                )
            )

        def __str__(self) -> str:
            return format_dict(self)

    def _init(self, **kwargs: Any) -> None:
        """Verify file size and read all records."""
        assert self._fh is not None
        if self._filesize % 472 or self._filesize // 472 > 1024:
            raise LfdFileError(self)
        records = []
        for rec in numpy.rec.fromfile(self._fh, self._record_t):
            number = int(rec['number'])
            if number == 0:
                continue
            if number > 25:
                warnings.warn('corrupted record')
                continue
            record = self.Record()
            record['number'] = number
            record['title'] = bytes2str(rec['title'][: rec['_title_len']])
            record['nanal'] = int(rec['nanal'])
            record['date'] = '{}.{}.{}'.format(*rec['date'])
            record['time'] = '{}:{}:{}'.format(*rec['time'])
            record['frequency'] = numpy.array(
                [float(f[:i].strip()) for i, f in rec['frequency'][:number]],
                dtype=numpy.float64,
            )
            record['phase'] = rec['phase'][:number] / 100.0
            record['modulation'] = rec['modulation'][:number] / 100.0
            record['deltap'] = rec['deltap'][:number] / 100.0
            record['deltam'] = rec['deltam'][:number] / 100.0
            records.append(record)
        self._records = records

    def _asarray(
        self,
        key: int = 0,
        **kwargs: Any,
    ) -> NDArray[numpy.float64]:
        """Return freq, phi, mod, dp, dm of selected record as NumPy array."""
        if self._records:
            return self._records[key].asarray()
        return numpy.empty((5, 0), numpy.float64)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Plot all phase and modulation vs log of frequency."""
        assert pyplot is not None and cycler is not None
        maxplots = 50
        colors: list[str] = []
        for c in (
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ):
            colors.extend((c, c))
        colors = cycler.cycler(color=colors)
        pyplot.subplots_adjust(bottom=0.12)
        # phase and modulation
        ax = pyplot.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
        ax.set_title(self._filename)
        ax.set_ylabel('Phase (\xb0) and Modulation (%)')
        ax.set_prop_cycle(colors)
        for rec in self._records[:maxplots]:
            ax.semilogx(rec['frequency'], rec['phase'], '+-')
            ax.semilogx(rec['frequency'], rec['modulation'], '.-')
        # delta
        ax = pyplot.subplot2grid((3, 1), (2, 0), sharex=ax)
        ax.set_ylabel('Delta Phase and Modulation')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_prop_cycle(colors)
        for rec in self._records[:maxplots]:
            ax.semilogx(rec['frequency'], rec['deltap'], '+-')
            ax.semilogx(rec['frequency'], rec['deltam'], '.-')

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        raise ValueError('GlobalsLif file does not contain image data')

    def _str(self) -> str | None:
        """Return string with information about file."""
        return f'records: {len(self._records)}'
        # return '\n '.join(f'Record {i}\n{format_dict(r, "  ", trim=0)}'
        #                   for i, r in enumerate(self._records))

    def __getitem__(self, key: int, /) -> Record:
        """Return selected record."""
        return self._records[key]

    def __len__(self) -> int:
        """Return number of records."""
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        """Return iterator over records."""
        return iter(self._records)


class GlobalsAscii(LfdFile):
    """Globals ASCII lifetime data.

    Globals ASCII files contain array and meta data of a single frequency
    domain lifetime measurement, stored as human readable ASCII string.
    Consecutive measurements are stored in separate files with increasing file
    extension numbers. The format is also used by ISS and FLOP software.

    The metadata can be accessed via dictionary `getitem` interface.
    Keys are lower case with spaces replaced by underscores.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with GlobalsAscii('FLOP.001') as f:
        ...     print(f['experiment'], f.asarray().shape)
        ...
        LIFETIME (5, 20)

    """

    _filemode = 'r'
    _fileencoding = 'cp1252'
    _filepattern = r'.*\.(\d){3}$'
    _figureargs = {'figsize': (6, 5)}

    _record: dict[str, Any]

    def _init(self, **kwargs: Any) -> None:
        """Read file and parse into dictionary and data array."""
        assert self._fh is not None
        if self._fh.read(5) != 'TITLE':
            raise LfdFileError(self)

        self._fh.seek(0)
        content = self._fh.read()
        self._fsize = len(content)
        # parse keys and values
        self._record = {}
        matches = re.findall(
            r'(.*?)(?:\((.*)\))?:\s(.*)', content, re.IGNORECASE
        )
        for key, unit, value in matches:
            key = key.lower().strip().replace(' ', '_')
            unit = unit.strip()
            value = astype(value.strip())
            self._record[key] = value
            if unit:
                self._record[key + '_unit'] = unit
        # extract data array
        match = re.search(
            r'DATA:(.*)[\r\n]([-+\.\d\s\r\n]*)ENDDATA', content, re.IGNORECASE
        )
        assert match is not None
        labels = tuple(d.strip().title() for d in match.group(1).split(', '))
        datastr = match.group(2)
        try:
            data = numpy.fromstring(datastr, dtype=numpy.float32, sep=' ')
            data = data.reshape((len(labels), -1), order='F')
        except ValueError:
            # try to reconstruct numbers not separated by spaces
            if self._record['experiment'] != 'LIFETIME':
                raise
            data_list: list[float] = []
            for line in datastr.splitlines():
                line = line.split('.')
                data_list.extend(
                    (
                        float(line[0][0:] + '.' + line[1][:7]),
                        float(line[1][7:] + '.' + line[2][:3]),
                        float(line[2][3:] + '.' + line[3][:3]),
                        float(line[3][3:] + '.' + line[4][:3]),
                        float(line[4][3:] + '.' + line[5]),
                    )
                )
            data = numpy.array(data_list, dtype=numpy.float32)
            data = data.reshape((len(labels), -1), order='F')
        self.__data = data
        self._record['data_shape'] = self.__data.shape
        self._record['data'] = labels
        self.close()

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return array data as NumPy array."""
        return self.__data.copy()

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Plot phase and modulation vs log of frequency."""
        assert pyplot is not None
        if self['experiment'] != 'LIFETIME':
            pyplot.title(f"Can not display {self['experiment']} data")
            return
        pyplot.subplots_adjust(bottom=0.12)
        data = self.asarray()
        # phase and modulation
        ax = pyplot.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
        ax.set_title(self._filename)
        ax.set_xlim([data[0][0], data[0][-1]])
        ax.semilogx(data[0], data[1], 'bx-', label='Phase (\xb0)')
        ax.semilogx(data[0], data[3] * 100, 'gx-', label='Modulation (%)')
        ax.xaxis.set_visible(False)
        ax.legend(loc='center left')
        # delta
        ax = pyplot.subplot2grid((3, 1), (2, 0), sharex=ax)
        ax.set_xlim([data[0][0], data[0][-1]])
        ax.semilogx(data[0], data[2], 'bx-')
        ax.semilogx(data[0], data[4] * 100, 'gx-')
        ax.set_xlabel('Frequency (MHz)')

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        raise ValueError('GlobalsAscii file does not contain image data')

    def _str(self) -> str:
        """Return string with information about file."""
        return format_dict(self._record)

    def __getitem__(self, key: str, /) -> Any:
        """Return value of key in record."""
        return self._record[key]


class VistaIfi(LfdFile):
    """VistaVision fluorescence intensity image.

    VistaVision IFI files contain multi-dimensional confocal intensity images.

    After a header of 256 bytes, the images are stored as float32 in CZYX
    order.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> f = VistaIfi('vista.ifi')
        >>> data = f.asarray()
        >>> f.axes
        'CYX'
        >>> print(f.header.dwelltime)
        0.1
        >>> data.shape
        (2, 128, 128)
        >>> f.totiff('_vista.ifi.tif')
        >>> f.close()
        >>> with TiffFile('_vista.ifi.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.ifi$'

    header: numpy.recarray[Any, Any]
    """File header."""

    _header_t = numpy.dtype(
        [
            # undocumented file header
            ('signature', 'S10'),  # 'VISTAIMAGE'
            ('version', 'u2'),
            ('channel_bits', 'u2'),
            ('dimensions', 'u2', 3),  # XYZ
            ('boundaries', 'f4', 6),
            ('dwelltime', 'f4'),
        ]
    )

    def _init(self, **kwargs: Any) -> None:
        """Read header and metadata from file."""
        assert self._fh is not None
        self.header = h = numpy.rec.fromfile(  # type: ignore[call-overload]
            self._fh, self._header_t, shape=1, byteorder='<'
        )[0]
        if h['signature'] != b'VISTAIMAGE':
            raise LfdFileError(self)
        if h['version'] != 4:
            logger().warning(
                f'unrecognized VistaIfi file version {h["version"]}'
            )
        self.shape = tuple(int(i) for i in reversed(h['dimensions']))
        if self.shape[0] == 1:
            self.shape = self.shape[1:]
            self.axes = 'YX'
        else:
            self.axes = 'ZYX'
        indices = self.channel_indices
        if len(indices) > 1:
            self.shape = (len(indices),) + self.shape
            self.axes = 'C' + self.axes
        self.dtype = numpy.dtype(numpy.float32)

    def __getitem__(self, key: str, /) -> Any:
        """Return header attribute."""
        return self.header[key]

    @property
    def channel_indices(self) -> tuple[int, ...]:
        """Indices of valid channels."""
        bits = self.header['channel_bits']
        return tuple(i for i in range(8) if bits & 2**i)

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return image data from file."""
        assert (
            self._fh is not None
            and self.shape is not None
            and self.dtype is not None
        )
        self._fh.seek(256)
        data = numpy.fromfile(self._fh, '<f4', product(self.shape))
        data.shape = self.shape
        return data

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, metadata={'axes': self.axes})
        data = self.asarray()
        tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return additional information about file."""
        assert self._header_t.names is not None
        return indent(
            'header:',
            *(
                f'{k}: {self.header[k]}'[:64]
                for k in self._header_t.names
                if k[0] != '_'
            ),
        )


class VistaIfli(LfdFile):
    """VistaVision fluorescence lifetime image.

    VistaVision IFLI files contain phasor images for several
    positions, wavelengths, time points, channels, slices, and frequencies
    from analog or digital frequency domain fluorescence lifetime measurements.
    After a version dependent header of 1024 bytes, the image is stored as
    a 9 dimensional arrays of float32 and shape
    (position, wavelength, time, channel, Z, Y, X, frequency, sample).
    The phasor array has three samples: the average intensity (DC) and the
    real (g) and imaginary (s) parts of the phasor.
    Additional metadata may be stored in an associated XML file.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> f = VistaIfli('vista.ifli')
        >>> data = f.asarray(memmap=True)
        >>> f.header['ModFrequency']
        (48000000.0, 96000000.0)
        >>> data.shape
        (1, 1, 1, 2, 1, 128, 128, 2, 3)
        >>> f.totiff('_vista.ifli.tif')
        >>> f.close()
        >>> with TiffFile('_vista.ifli.tif') as f:
        ...     assert_array_equal(f.asarray()[0, 0], data[..., 0, 0])
        ...

    """

    _filepattern = r'.*\.ifli$'
    _figureargs = {'figsize': (8, 7.5)}

    header: dict[str, Any]
    """Metadata from file header."""

    offsets: dict[str, Any]
    """Data offsets from file header."""

    def _init(self, **kwargs: Any) -> None:
        """Read header and metadata from file."""
        assert self._fh is not None
        fh = self._fh

        signature = fh.read(13)
        if signature[:12] != b'VistaFLImage':
            raise LfdFileError(self)

        header: dict[str, Any] = {}
        header['Version'] = version = signature[12]

        fields = {}
        if version >= 16:
            fields['SpectralLifetime'] = '?'
            fields['SpectralPhasor'] = '?'
            fields['SpectralIntensity'] = '?'
        if version >= 15:
            fields['HistogramResolution'] = 'I'
        fields['ChannelNumbers'] = 'I' if version >= 14 else 'B'
        fields['CompressionType'] = 'B'
        fields['XYZCTsize'] = '5H'
        fields['XYZCTrange'] = '6f'  # x0, x1, y0, y1, z0, z1
        fields['CordinateUnit'] = 'B'  # unknown, px, um, mm
        fields['PixelTime'] = 'f'
        fields['PixelIntervalTime'] = 'f'
        fields['LineIntervalTime'] = 'f'
        fields['FrameIntervalTime'] = 'f'
        fields['ModFrequencyCount'] = 'i'
        fields['CrossCorrelationFrequency'] = 'f'
        fields['FrameRepeat'] = 'H'
        read_record(header, fields, fh)

        if header['FrameRepeat'] < 1:
            header['FrameRepeat'] = 1

        sizex, sizey, sizez, sizec, sizet = header['XYZCTsize']
        sizef = header['ModFrequencyCount']
        sizer = 1
        sizee = 1

        fh.seek(256)
        offsets: dict[str, int] = {}
        inttype = 'Q' if version >= 7 else 'I'
        fields = {
            'PhasorPixelData': inttype,
            'TimeTags': inttype,
            'ModFrequency': inttype,
            'RefLifetime': inttype,
            'RefDCPhasor': inttype,
            'PhasorStdErrPixelData': inttype,
            'AnalysisData': inttype,
            'PseriesInfo': inttype,
            'SpectrumInfo': inttype,
            'TimeSeriesExternalClockFrequencies': inttype,
        }
        if version >= 11:
            fields['PlimInfo'] = inttype
        if version >= 12:
            fields['FrameScanPlaneInfo'] = inttype
        fields['CommentsInfo'] = inttype
        read_record(offsets, fields, fh)

        if offsets['PhasorPixelData'] == 0:
            offsets['PhasorPixelData'] = 256
        if offsets['ModFrequency'] == 0:
            offsets['ModFrequency'] = 70

        if header['CompressionType'] != 0:
            raise NotImplementedError(
                f'VistaIfli compression {header["CompressionType"]} '
                'not supported'
            )

        if sizet > 0:
            if offsets['TimeTags'] > 0:
                fh.seek(offsets['TimeTags'])
                header['TimeTags'] = struct.unpack(
                    f'<{sizet}f', fh.read(sizet * 4)
                )
            if offsets['TimeSeriesExternalClockFrequencies'] > 0:
                fh.seek(offsets['TimeSeriesExternalClockFrequencies'])
                header['TimeSeriesExternalClockFrequencies'] = struct.unpack(
                    f'<{sizet}f', fh.read(sizet * 4)
                )
        if offsets['PseriesInfo'] > 0:
            fh.seek(offsets['PseriesInfo'])
            sizer, header['PseriesInfoMode'] = struct.unpack('<HH', fh.read(4))
            header['PseriesInfo'] = struct.unpack(
                f'<{sizer * 3}f', fh.read(sizer * 12)
            )
        if version >= 13 and offsets['SpectrumInfo'] > 0:
            fh.seek(offsets['SpectrumInfo'])
            sizee = struct.unpack('<H', fh.read(2))[0]
            header['SpectrumInfo'] = struct.unpack(
                f'<{sizee}f', fh.read(sizee * 4)
            )
        if version >= 11 and offsets['PlimInfo'] > 0:
            fh.seek(offsets['PlimInfo'])
            header['PlimInfo'] = struct.unpack('<f', fh.read(4))[0]
        if version >= 12 and offsets['FrameScanPlaneInfo'] > 0:
            fh.seek(offsets['FrameScanPlaneInfo'])
            header['FrameScanPlaneInfo'] = struct.unpack('<f', fh.read(4))
        if version >= 9 and offsets['CommentsInfo'] > 0:
            fh.seek(offsets['CommentsInfo'])
            size, hdrlen = uleb128(fh.read(4))
            if size > 0:
                fh.seek(offsets['CommentsInfo'] + hdrlen)
                try:
                    header['Comments'] = fh.read(size).decode()
                except UnicodeDecodeError:
                    logger().warning('failed to read CommentsInfo')
        if offsets['ModFrequency'] > 0:
            fh.seek(offsets['ModFrequency'])
            header['ModFrequency'] = struct.unpack(
                f'<{sizef}f', fh.read(sizef * 4)
            )

        offset = offsets['RefLifetime']
        if offset == 0:
            offset = offsets['ModFrequency'] + sizef * 4 + 4
        fh.seek(offset)
        header['RefLifetime'] = struct.unpack(f'<{sizec}f', fh.read(sizec * 4))
        if version >= 10:
            header['RefLifetimeFrac'] = struct.unpack(
                f'<{sizec}f', fh.read(sizec * 4)
            )
            header['RefLifetime2'] = struct.unpack(
                f'<{sizec}f', fh.read(sizec * 4)
            )
        else:
            header['RefLifetimeFrac'] = (1.0,) * sizec
            header['RefLifetime2'] = (0.0,) * sizec

        offset = offsets['RefDCPhasor']
        if offset == 0:
            offset = offsets['ModFrequency'] + sizef * 4 + 8
        fh.seek(offset)
        header['RefDCPhasor'] = numpy.frombuffer(
            fh.read(sizec * sizef * 3 * 4), numpy.float32
        ).reshape(sizec, sizef, 3)

        if sizer > 1 or sizee > 1:
            # TODO: Support for multi-positional or spectral VistaIfli
            logger().warning(
                'Support for multi-positional or spectral VistaIfli files is '
                'experimental. Please submit a sample file.'
            )

        self.axes = 'RETCZYXFS'
        self.shape = (
            sizer,
            sizee,
            sizet,
            sizec,
            sizez,
            sizey,
            sizex,
            sizef,
            3,
        )
        self.dtype = numpy.dtype(numpy.float32)
        self.header = header
        self.offsets = offsets

    @property
    def channel_indices(self) -> tuple[int, ...]:
        """Indices of valid channels."""
        assert self.shape is not None
        bits = self.header['ChannelNumbers']
        return tuple(i for i in range(self.shape[3]) if bits & 2**i)

    def _asarray(self, *, memmap: bool = False, **kwargs: Any) -> NDArray[Any]:
        """Return average intensity and phasor coordinates from file.

        The returned array is of shape (position, wavelength, time, channel,
        Z, Y, X, frequency, sample) and type float32.
        The three samples are the average intensity (DC) and the real (g) and
        imaginary (s) parts of the phasor.

        """
        assert self._fh is not None and self.shape is not None
        offset = self.offsets['PhasorPixelData']
        if memmap:
            return numpy.memmap(
                self._fh,
                dtype='<f4',
                mode='r',
                offset=offset,
                shape=self.shape,
                **kwargs,
            )
        self._fh.seek(offset)
        phasor = numpy.fromfile(self._fh, '<f4', product(self.shape))
        phasor.shape = self.shape
        return phasor

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in file."""
        assert pyplot is not None
        data = self.asarray()
        data[..., 0] /= numpy.max(data[..., 0])
        data = numpy.moveaxis(data, -1, 0)
        data = numpy.moveaxis(data, -1, -3)
        imshow(
            data,
            figure=figure,
            title=self._filename,
            vmin=-1,
            vmax=1,
            photometric='MINISBLACK',
        )

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False)
        data = self.asarray()
        data = numpy.moveaxis(data, -2, 0)
        data = numpy.moveaxis(data, -1, 0)
        tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return additional information about file."""
        return indent(
            'header:',
            *(
                f'{name}: {value}'
                for name, value in self.header.items()
                if name != 'Comments'
            ),
            *(
                f'{name}Offset: {value}'
                for name, value in self.offsets.items()
            ),
            (
                f'Comments:\n{self.header["Comments"]}'
                if 'Comments' in self.header
                else ''
            ),
        )


class FlimfastFlif(LfdFile):
    """FlimFast fluorescence lifetime image.

    FlimFast FLIF files contain camera images and metadata of frequency-domain
    fluorescence lifetime measurements.
    A 640-byte header is followed by a variable number of uint16 images,
    each preceded by a 64-byte record.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> f = FlimfastFlif('flimfast.flif')
        >>> data = f.asarray()
        >>> float(f.header.frequency)
        80.652...
        >>> float(f.records['phase'][31])
        348.75
        >>> int(data[31, 219, 299])
        366
        >>> f.totiff('_flimfast.flif.tif')
        >>> f.close()
        >>> with TiffFile('_flimfast.flif.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.flif$'
    _figureargs = {'figsize': (6, 7)}

    header: numpy.recarray[Any, Any]
    """File header."""

    records: numpy.recarray[Any, Any]
    """Frame header."""

    _header_t = numpy.dtype(
        [
            ('magic', 'S8'),  # '\211FLF\r\n0\n'
            ('creator', 'S120'),
            ('date', 'S32'),
            ('comments', 'S351'),
            ('_', 'u1'),
            ('fileprec', '<u2'),
            ('records', '<u2'),
            ('phases', '<i4'),
            ('width', '<i4'),
            ('height', '<i4'),
            ('dataprec', '<i4'),
            ('background', '<i4'),
            ('camframes', '<i4'),
            ('cambin', '<i4'),
            ('roileft', '<i4'),
            ('roitop', '<i4'),
            ('frequency', '<f4'),
            ('ref_tauphase', '<f4'),
            ('measured_phase', '<f4'),
            ('measured_mod', '<f4'),
            ('start', '<u4'),
            ('duration', '<u4'),
            ('phaseoffset', '<f4'),
            ('ref_taumod', '<f4'),
        ]
    )  # ('padding', 'i4', 14)

    _record_t = numpy.dtype(
        [
            ('index', '<i4'),
            ('order', '<i4'),
            ('phase', '<f4'),
            ('integrated', '<i4'),
            ('time', '<u4'),
        ]
    )  # ('padding', 'u4', 11)

    def _init(self, **kwargs: Any) -> None:
        """Read header and record metadata from file."""
        assert self._fh is not None
        if not self._fh.read(8) == b'\211FLF\r\n0\n':
            raise LfdFileError(self)
        self._fh.seek(0)
        h = self.header = numpy.rec.fromfile(
            # type: ignore[call-overload]
            self._fh,
            self._header_t,
            shape=1,
            byteorder='<',
        )[0]
        h['creator'] = stripnull(h.creator)
        h['date'] = stripnull(h.date)
        h['comments'] = stripnull(h.comments)
        if not (
            h.magic == b'\211FLF\r\n0\n'
            and 1 <= h.width <= 4096
            and 1 <= h.height <= 4096
            and 1 <= h.phases <= 1024
            and 2 <= h.records <= 1024
        ):
            raise LfdFileError(self)
        self.records = numpy.recarray((h.phases,), self._record_t)
        stride = 11 * 4 + 2 * h.width * h.height  # padding + image
        self._fh.seek(14 * 4, 1)  # header padding
        for i in range(h.phases):
            self.records[i] = numpy.rec.fromfile(
                # type: ignore[call-overload]
                self._fh,
                self._record_t,
                shape=1,
                byteorder='<',
            )[0]
            self._fh.seek(stride, 1)
        self.shape = int(h.records), int(h.height), int(h.width)
        self.dtype = numpy.dtype('<u2')
        self.axes = 'PYX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.uint16]:
        """Return images as uint16 array of shape (records, height, width)."""
        assert (
            self._fh is not None
            and self.shape is not None
            and self.dtype is not None
        )
        p, h, w = self.shape
        data = numpy.empty((p, h * w), numpy.uint16)
        self._fh.seek(640 + 64)
        for i in range(p):
            data[i] = numpy.fromfile(self._fh, self.dtype, h * w)
            self._fh.seek(64, 1)
        data.shape = self.shape
        return data

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write phase images and metadata to TIFF file."""
        metadata = {}
        dtypes = {
            'f': float,
            'i': int,
            'u': int,
            'S': lambda x: str(x, 'latin-1'),
        }
        for name, dtype in self.header.dtype.fields.items():
            if name not in {'_', 'magic'}:
                dtype = dtypes[dtype[0].kind]
                metadata[name] = dtype(self.header[name])
        for name, dtype in self.records.dtype.fields.items():
            dtype = dtypes[dtype[0].kind]
            metadata[name] = list(dtype(i) for i in self.records[name])
        update_kwargs(kwargs, contiguous=True, metadata=metadata)
        for data in self.asarray():
            tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return file header as string."""
        assert self._header_t.names is not None
        return '\n'.join(
            (f'{name}: {getattr(self.header, name)}')[:79]
            for name in self._header_t.names[1:]
            if not name.startswith('_')
        )


class FlimageBin(LfdFile):
    """FLImage fluorescence lifetime image.

    FLImage BIN files contain referenced fluorescence lifetime image data
    from frequency-domain measurements.
    Three 300x220 big-endian float32 images are stored in separate files:
    intensity (``.int.bin``), phase (``.phi.bin``) and modulation
    (``.mod.bin``), respectively single apparent lifetimes from phase
    (``.tph.bin``) and modulation (``.tmd.bin``).
    Phase values are in degrees, modulation in percent.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with FlimageBin('flimage.int.bin') as f:
        ...     data = f.asarray()
        ...     f.totiff('_flimage.int.bin.tif')
        ...     print(f.shape, data[:, 219, 299])
        ...
        (3, 220, 300) [1.23 111.8 36.93]
        >>> with TiffFile('_flimage.int.bin.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(int|mod|phi|tmd|tph)\.bin$'
    _figureargs = {'figsize': (6, 7)}

    def _init(self, **kwargs: Any) -> None:
        """Verify file size is 264000."""
        if self._filesize != 264000:
            raise LfdFileError(self)
        self.shape = 220, 300
        self.dtype = numpy.dtype('>f4')
        self.axes = 'YX'

    def _components(self) -> list[tuple[str, str]]:
        """Return possible names of component files."""
        return [
            (c, self._filename[:-7] + c + '.bin')
            for c in ('int', 'phi', 'mod', 'tph', 'tmd')
        ]

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return images as float32 array of shape (220, 300)."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in files."""
        assert pyplot is not None
        update_kwargs(kwargs, cmap='viridis')
        images = self.asarray()
        pyplot.subplots_adjust(bottom=0.03, top=0.97, hspace=0.1, wspace=0.1)
        axes = [
            pyplot.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2),
            pyplot.subplot2grid((3, 2), (2, 0)),
            pyplot.subplot2grid((3, 2), (2, 1)),
        ]
        name = [name for name, _ in self.components]
        for i, (img, ax, title) in enumerate(
            zip(
                images,
                axes,
                (self._filename + ' - ' + name[0], name[1], name[2]),
            )
        ):
            ax.set_title(title)
            if i > 0:
                ax.set_axis_off()
            ax.imshow(img, **kwargs)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        tif.write(self.asarray(), **kwargs)


class FlieOut(LfdFile):
    """Flie fluorescence lifetime image.

    Flie OUT files contain referenced fluorescence lifetime image data
    from frequency-domain measurements.
    Three 300x220 big-endian float32 images are stored in separate files:
    intensity (``off_*.out``), phase (``phi_*.out``), and modulation
    (``mod_*.out``). Phase values are in degrees, modulation in percent.
    No metadata are available.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with FlieOut('off_flie.out') as f:
        ...     data = f.asarray()
        ...     f.totiff('_off_flie.out.tif')
        ...     print(f.shape, data[:, 219, 299])
        ...
        (3, 220, 300) [91.85 28.24 69.03]
        >>> with TiffFile('_off_flie.out.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'(off|phi|mod)_.*\.out$'
    _figureargs = {'figsize': (6, 7)}

    def _init(self, **kwargs: Any) -> None:
        """Verify file size is 264000."""
        if self._filesize != 264000:
            raise LfdFileError(self)
        self.shape = 220, 300
        self.dtype = numpy.dtype('>f4')
        self.axes = 'YX'

    def _components(self) -> list[tuple[str, str]]:
        """Return possible names of component files."""
        return [(c, c + self._filename[3:]) for c in ('Off', 'Phi', 'Mod')]

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float32]:
        """Return image data as float32 array of shape (220, 300)."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        tif.write(self.asarray(), **kwargs)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in files."""
        assert pyplot is not None
        update_kwargs(kwargs, cmap='viridis')
        images = self.asarray()
        pyplot.subplots_adjust(bottom=0.03, top=0.97, hspace=0.1, wspace=0.1)
        axes = [
            pyplot.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2),
            pyplot.subplot2grid((3, 2), (2, 0)),
            pyplot.subplot2grid((3, 2), (2, 1)),
        ]
        for i, (img, ax, title) in enumerate(
            zip(images, axes, (self._filename + ' - Off', 'Phi', 'Mod'))
        ):
            ax.set_title(title)
            if i > 0:
                ax.set_axis_off()
            ax.imshow(img, **kwargs)


class FliezI16(LfdFile):
    """FLIez integer image.

    FLIez I16 files contain camera images, usually for one phase cycle of
    frequency-domain fluorescence lifetime measurements.
    Several 256x256 uint16 intensity images is stored consecutively.
    No metadata are available.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with FliezI16('fliez.i16') as f:
        ...     data = f.asarray()
        ...     f.totiff('_fliez.i16.tif')
        ...     print(f.shape, data[::8, 108, 104])
        ...
        (32, 256, 256) [401 538 220 297]
        >>> with TiffFile('_fliez.i16.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.i16$'

    def _init(self, **kwargs: Any) -> None:
        """Verify file size is 128 KB."""
        if self._filesize % 131072:
            raise LfdFileError(self)
        self.shape = int(self._filesize // 131072), 256, 256
        self.dtype = numpy.dtype('<u2')
        self.axes = 'IYX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.uint16]:
        """Return images as uint16 array of shape (-1, 256, 256)."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        tif.write(self.asarray(), **kwargs)


class FliezDb2(LfdFile):
    """FLIez double image.

    FLIez DB2 files contain a sequence of images from fluorescence lifetime
    measurements. The modality of the data stored in different files varies:
    phase intensities, average intensities, phase or modulation.
    After a header specifying the 3D data shape, images are stored
    consecutively as float64. No other metadata are available.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with FliezDb2('fliez.db2') as f:
        ...     data = f.asarray()
        ...     f.totiff('_fliez.db2.tif')
        ...     print(f.shape, data[8, 108, 104])
        ...
        (32, 256, 256) 234.0
        >>> with TiffFile('_fliez.db2.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.db2$'

    def _init(self, **kwargs: Any) -> None:
        """Read data shape and verify file size."""
        assert self._fh is not None
        shape = struct.unpack('<iii', self._fh.read(12))
        if self._filesize - 12 != product(shape) * 8:
            raise LfdFileError(self)
        self.shape = shape[::-1]
        self.dtype = numpy.dtype('<f8')
        self.axes = 'IYX'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.float64]:
        """Return images as 3D array of float64."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        tif.write(self.asarray(), **kwargs)


class BioradPic(LfdFile):
    """Bio-Rad(tm) multi-dimensional data.

    Bio-Rad PIC files contain single-channel volume data or multi-channel
    images.

    Image data in uint8 or uint16 format are stored after a 76-byte header.
    Additional metadata are stored after the image data as 96-byte records
    ("notes").

    No official file format specification is available.
    The header structure was obtained from
    https://forums.ni.com/ni/attachments/ni/200/7567/1/file%20format.pdf
    This implementation does not currently handle multi-file data.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with BioradPic('biorad.pic') as f:
        ...     data = f.asarray()
        ...     f.totiff('_biorad.pic.tif')
        ...     print(f.shape, data[78, 255, 255])
        ...
        (79, 256, 256) 8
        >>> with TiffFile('_biorad.pic.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.pic$'
    _filesizemin = 76
    _figureargs = {'figsize': (8, 6)}

    header: dict[str, Any]
    """Select values of header structure."""

    notes: list[tuple[str, int, int, int, int]]
    """Additional information about images."""

    origin: tuple[float, ...]
    """Position of first pixel or voxel in micrometer."""

    spacing: tuple[float, ...]
    """Spacing of pixels or voxels in micrometer."""

    def _init(self, **kwargs: Any) -> None:
        """Read header and validate file Id."""
        assert self._fh is not None
        (
            nx,
            ny,
            npic,
            ramp1_min,
            ramp1_max,
            notes,
            byte_format,
            image_number,
            name,
            merged,
            color1,
            file_id,
            ramp2_min,
            ramp2_max,
            color2,
            edited,
            lens,
            mag_factor,
        ) = struct.unpack('<hhhhhIhh32shHHhhHhhf6x', self._fh.read(76))
        if file_id != 12345:
            raise LfdFileError(self)
        self.header = dict(
            name=bytes2str(name).strip(),
            ramp1_min=ramp1_min,
            ramp1_max=ramp1_max,
            color1=color1,
            ramp2_min=ramp2_min,
            ramp2_max=ramp2_max,
            color2=color2,
            image_number=image_number,
            lens=lens,
            mag_factor=mag_factor,
            edited=edited,
            merged=merged,
        )

        self.dtype = numpy.dtype('<u1' if byte_format else '<u2')
        if npic > 1:
            self.shape = npic, ny, nx
            self.axes = 'IYX'
        else:
            self.shape = ny, nx
            self.axes = 'YX'
        self.notes = []
        self.spacing = ()
        self.origin = ()
        if notes != 0:
            try:
                self.notes, self.spacing, self.origin = self._notes()
            except Exception as exc:
                warnings.warn(f'failed to read PIC notes: {exc}')
        ndims = len(self.spacing)
        if npic > 1 and ndims in {2, 3}:
            self.axes = 'ZYX' if ndims == 3 else 'CYX'

    def _notes(
        self,
    ) -> tuple[
        list[tuple[str, int, int, int, int]],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        """Return metadata from notes records in file."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        pos = 76 + product(self.shape) * self.dtype.itemsize
        self._fh.seek(pos)
        if self._fh.tell() != pos:
            raise LfdFileError(self, 'file is too small')
        more = True
        spacing = []
        origin = []
        notes = []
        while more:
            level, more, notetype, x, y, note = struct.unpack(
                '<hi4xhhh80s', self._fh.read(96)
            )
            note = bytes2str(note).strip()
            # TODO: parse notes to dict
            notes.append((note, notetype, level, x, y))
            if note[:5] == 'AXIS_':
                index, kind, ori, res = note[5:].split(None, 4)[:4]
                if kind == '001':
                    spacing.append(float(res))
                    origin.append(float(ori))
        return notes, tuple(spacing), tuple(origin)

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.uint8 | numpy.uint16]:
        """Return image data as array."""
        assert (
            self._fh is not None
            and self.shape is not None
            and self.dtype is not None
        )
        self._fh.seek(76)
        data = numpy.fromfile(self._fh, self.dtype, product(self.shape))
        return data.reshape(*self.shape)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in file."""
        data = self._asarray()
        imshow(
            data, figure=figure, title=self._filename, photometric='MINISBLACK'
        )

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        update_kwargs(
            kwargs,
            metadata={
                'axes': self.axes,
                'spacing': self.spacing,
                'origin': self.origin,
            },
        )
        tif.write(self.asarray(), **kwargs)

    def _str(self) -> str | None:
        """Return properties as string."""
        return f'spacing: {self.spacing}\norigin: {self.origin}\n' + indent(
            'header:', format_dict(self.header, prefix='')
        )


def bioradpic_write(
    filename: os.PathLike[Any] | str,
    data: ArrayLike,
    /,
    *,
    axis: str = 'Z',
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    name: str | None = None,
    lens: int = 1,
    mag_factor: float = 1.0,
    image_number: int = 0,
    merged: int = 0,
    edited: int = 0,
    ramp1_min: int = 0,
    ramp1_max: int = 0,
    color1: int = 0,
    ramp2_min: int = 0,
    ramp2_max: int = 0,
    color2: int = 0,
) -> None:
    """Write volume or multi-channel data to Bio-Rad(tm) PIC formatted file.

    This implementation does not currently allow writing multi-file datasets,
    advanced metadata, or color palettes.

    Parameters:
        filename:
            Name of file to write.
        data:
            Data to write.
            Must be two or three-dimensional of type uint18 or uint16.
        spacing, origin:
            Position and spacing of pixel or voxel in micrometer.
        name, lens, mag_factor, image_number, merged, edited, \
        ramp1_min, ramp1_max, color1, ramp2_min, ramp2_max, color2:
            Refer to the Biorad PIC header documentation.

    Examples:
        >>> data = numpy.arange(1000000).reshape(100, 100, 100).astype('u1')
        >>> bioradpic_write('_test.pic', data)
        >>> with BioradPic('_test.pic') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """
    data = numpy.asarray(data)
    if data.ndim not in {2, 3}:
        raise ValueError('data must be 2 or 3 dimensional')
    if data.dtype.char not in 'BH':
        raise ValueError('data type must be uint8 or uint16')
    if data.ndim == 2:
        data.shape = 1, data.shape[0], data.shape[1]
    if name is None:
        name = os.path.split(filename)[-1]
    header = struct.pack(
        '<hhhhhIhh32shHHhhHhhf6b',
        data.shape[2],
        data.shape[1],
        data.shape[0],
        ramp1_min,
        ramp1_max,
        1,
        1 if data.dtype.char == 'B' else 0,
        image_number,
        name[:31].encode('latin1'),
        merged,
        color1,
        12345,
        ramp2_min,
        ramp2_max,
        color2,
        edited,
        lens,
        mag_factor,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    axes = (2, 3, 4, 9) if axis == 'Z' else (2, 3, 4)
    ndim = len(axes) - 1
    if origin is None:
        origin = [0.0] * ndim
    if spacing is None:
        spacing = [1.0] * ndim
    notes = [
        struct.pack(
            '<hi4bhhh80s',
            -1,
            1,
            0,
            0,
            0,
            0,
            20,
            0,
            0,
            b'AXIS_%i 001 %.6e %.6e microns'
            % (axes[i], origin[i], spacing[i]),
        )
        for i in range(ndim)
    ]
    notes.append(
        struct.pack(
            '<hi4bhhh80s',
            -1,
            0,
            0,
            0,
            0,
            0,
            20,
            0,
            0,
            b'AXIS_%i 011 0.000000e+00 1.000000e+00 RGB channel' % axes[-1],
        )
    )
    with open(filename, 'wb') as fh:
        fh.write(header)
        data.tofile(fh)
        fh.write(b''.join(notes))


class Ccp4Map(LfdFile):
    """CCP4 volume data.

    CCP4 MAP files contain 3D volume data. It is used by the Electron
    Microscopy Data Bank to store electron density maps.

    - <http://emdatabank.org/mapformat.html>
    - <http://www.ccp4.ac.uk/html/maplib.html>
    - <ftp://ftp.wwpdb.org/pub/emdb/doc/map_format/EMDB_mapFormat_v1.0.pdf>

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with Ccp4Map('ccp4.map') as f:
        ...     data = f.asarray()
        ...     f.totiff('_ccp4.map.tif', compression='zlib')
        ...     print(f.shape, data[100, 100, 100])
        ...
        (256, 256, 256) 1.0
        >>> with TiffFile('_ccp4.map.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(map|ccp4)$'
    _filesizemin = 1024 + 80

    start: tuple[int, int, int]
    """Position of first section, row, and column (voxel grid units)."""

    cell_interval: tuple[int, int, int]
    """Intervals per unit cell repeat along Z, Y, X."""

    cell_length: tuple[float, float, float]
    """Unit Cell repeats along Z, Y, X in Angstroms."""

    cell_angle: tuple[float, float, float]
    """Unit Cell angles (alpha, beta, gamma) in degrees."""

    map_src: tuple[int, int, int]
    """Relationship of Z, Y, X axes to sections, rows, columns."""

    skew_translation: NDArray[numpy.float64] | None
    """Translation vector."""

    skew_matrix: NDArray[numpy.float64] | None
    """Skew matrix."""

    symboltable: list[bytes]
    """Symmetry records as defined in International Tables."""

    labels: list[bytes]
    """Column labels."""

    density_min: float
    """Minimum density."""

    density_max: float
    """Maximum density."""

    density_mean: float
    """Average density."""

    density_rms: float
    """RMS deviation of map from mean density."""

    spacegoup: int
    """IUCr space group number (1-230)."""

    _dtypes: dict[int, str] = {0: 'i1', 1: 'i2', 2: 'f4', 4: 'q8', 5: 'i1'}

    def _init(self, **kwargs: Any) -> None:
        """Read CCP4 file header and symboltable."""
        assert self._fh is not None
        header = self._fh.read(1024)
        if header[208:212] not in {b'MAP ', b'PAM\x00', b'MAP\x00'}:
            raise LfdFileError(self, f' {header[:32]}')
        try:
            (
                nc,
                nr,
                ns,
                mode,  # data type
                ncstart,
                nrstart,
                nsstart,
                nx,
                ny,
                nz,
                x_length,
                y_length,
                z_length,
                alpha,
                beta,
                gamma,
                mapc,
                mapr,
                maps,
                self.density_min,
                self.density_max,
                self.density_mean,
                self.spacegoup,
                nsymbt,  # number of bytes used for storing symmetry operators
                skew_matrix_flag,
                S11,
                S12,
                S13,
                S21,
                S22,
                S23,
                S31,
                S32,
                S33,
                T1,
                T2,
                T3,
                # extra,
                map_,  # b'MAP '
                machst,  # machine stamp,
                self.density_rms,
                nlabl,  # number of labels used
                L0,
                L1,
                L2,
                L3,
                L4,
                L5,
                L6,
                L7,
                L8,
                L9,
            ) = struct.unpack(
                '3ii3i3i3f3f3i3fiii9f3f60x4s4sfi'
                '80s80s80s80s80s80s80s80s80s80s',
                header,
            )
        except struct.error as exc:
            raise LfdFileError(self) from exc
        try:
            # machst = header[212:216]
            byteorder = {b'DA\x00\x00': '<', b'\x11\x11\x00\x00': '>'}[machst]
        except KeyError:
            byteorder = '='
            warnings.warn(f'Ccp4Map: unknown machine stamp: {machst}')
        try:
            self.dtype = numpy.dtype(byteorder + Ccp4Map._dtypes[mode])
        except KeyError as exc:
            raise LfdFileError(self, f'unknown mode: {mode}') from exc
        self.shape = ns, nr, nc
        self.start = nsstart, nrstart, ncstart
        self.cell_interval = nz, ny, nx
        self.cell_length = z_length, y_length, x_length
        self.cell_angle = alpha, beta, gamma
        self.map_src = maps, mapr, mapc
        if skew_matrix_flag != 0:
            self.skew_translation = numpy.array([T1, T2, T3], numpy.float64)
            self.skew_matrix = numpy.array(
                [[S11, S12, S13], [S21, S22, S23], [S31, S32, S33]],
                numpy.float64,
            )  # .T?
        else:
            self.skew_translation = None
            self.skew_matrix = None
        if 0 <= nlabl <= 10:
            self.labels = [
                stripnull(lbl)
                for lbl in (L0, L1, L2, L3, L4, L5, L6, L7, L8, L9)[:nlabl]
            ]
        else:
            self.labels = []
        if nsymbt < 0 or nsymbt % 80:
            raise LfdFileError(self, f'invalid symbol table size: {nsymbt}')
        self.symboltable = [
            stripnull(self._fh.read(80)) for _ in range(nsymbt // 80)
        ]
        self.axes = 'ZYX'

    def _asarray(self, *, memmap: bool = False, **kwargs: Any) -> NDArray[Any]:
        """Return volume data as NumPy array.

        Parameters:
            memmap: If True, use `numpy.memmap` to read array.

        """
        assert self._fh is not None
        assert self.shape is not None
        assert self.dtype is not None
        if memmap:
            return numpy.memmap(  # type: ignore[no-any-return, call-overload]
                self._fh,
                dtype=self.dtype,
                mode='r',
                offset=self._pos,
                shape=self.shape,
            )
        data = numpy.fromfile(self._fh, self.dtype, product(self.shape))
        return data.reshape(self.shape)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        tif.write(self.asarray(), **kwargs)

    def _str(self) -> str | None:
        """Return additional information about file."""
        return f'cell length: {self.cell_length}'


def ccp4map_write(
    filename: os.PathLike[Any] | str,
    data: ArrayLike,
    /,
    *,
    start: Sequence[int] = (0, 0, 0),
    cell_interval: Sequence[int] | None = None,
    cell_length: Sequence[float] | None = None,
    cell_angle: Sequence[int] = (90, 90, 90),
    map_src: Sequence[int] = (3, 2, 1),
    density: Sequence[float] | None = None,
    density_rms: int = 0,
    spacegroup: int = 1,
    skew_matrix: ArrayLike | None = None,
    skew_translation: ArrayLike | None = None,
    symboltable: bytes = b'',
    labels: Sequence[bytes] = (b'Created by lfdfiles.py',),
) -> None:
    """Write 3D volume data to CCP4 MAP formatted file.

    Parameters:
        filename:
            Name of file to write.
        data:
            Volume data to write.
        start, cell_interval, cell_length, cell_angle, map_src, density, \
        density_rms, spacegroup, skew_matrix, skew_translation, \
        symboltable, labels:
            See :py:class:`Ccp4Map`.

    Examples:
        >>> data = numpy.arange(1000000).reshape(100, 100, 100).astype('f4')
        >>> ccp4map_write('_test.ccp4', data)
        >>> with Ccp4Map('_test.ccp4') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """
    data = numpy.asarray(data)
    if data.ndim != 3:
        raise ValueError('data must be 3 dimensional')
    try:
        mode = {'i1': 0, 'i2': 1, 'f4': 2, 'q8': 4}[data.dtype.str[-2:]]
    except KeyError as exc:
        raise ValueError('dtype not supported by MAP format') from exc
    if cell_interval is None:
        cell_interval = data.shape
    if cell_length is None:
        cell_length = data.shape  # ?
    if density is None:
        density = numpy.min(data), numpy.max(data), numpy.mean(data)
    if skew_matrix is None or skew_translation is None:
        skew_matrix_flag = 0
        S = numpy.zeros((3, 3))
        T = numpy.zeros(3)
    else:
        skew_matrix_flag = 1
        S = numpy.array(skew_matrix)
        T = numpy.array(skew_translation)
    assert S.shape == (3, 3)
    assert T.shape == (3,)
    labels = list(labels)[:10] if labels else []
    nlabl = len(labels)
    labels = [i[:79] for i in labels]
    labels.extend(b'' for _ in range(10 - nlabl))

    header = struct.pack(
        '3ii3i3i3f3f3i3fiii9f3f60x4s4sfi80s80s80s80s80s80s80s80s80s80s',
        data.shape[2],
        data.shape[1],
        data.shape[0],
        mode,
        start[2],
        start[1],
        start[0],
        cell_interval[2],
        cell_interval[1],
        cell_interval[0],
        cell_length[2],
        cell_length[1],
        cell_length[0],
        cell_angle[0],
        cell_angle[1],
        cell_angle[2],
        map_src[2],
        map_src[1],
        map_src[0],
        density[0],
        density[1],
        density[2],
        spacegroup,
        len(symboltable),
        skew_matrix_flag,
        S[0, 0],
        S[0, 1],
        S[0, 2],
        S[1, 0],
        S[1, 1],
        S[1, 2],
        S[2, 0],
        S[2, 1],
        S[2, 2],
        T[0],
        T[1],
        T[2],
        # extra,
        b'MAP ',
        {'little': b'DA\x00\x00', 'big': b'\x11\x11\x00\x00'}[sys.byteorder],
        density_rms,
        len(labels),
        labels[0],
        labels[1],
        labels[2],
        labels[3],
        labels[4],
        labels[5],
        labels[6],
        labels[7],
        labels[8],
        labels[9],
    )
    with open(filename, 'wb') as fh:
        fh.write(header)
        fh.write(symboltable)
        data.tofile(fh)


class Vaa3dRaw(LfdFile):
    """Vaa3D multi-channel volume data.

    Vaa3D RAW files contain 4D CZYX multi-channel volume data.

    The data is stored C-contiguously as uint8, uint16 or float32 in little
    or big-endian byte order, after a header defining data type, endianness,
    and shape.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with Vaa3dRaw('vaa3d.v3draw') as f:
        ...     data = f.asarray()
        ...     f.totiff('_vaa3d.v3draw.tif')
        ...     print(f.shape, data[2, 100, 100, 100])
        ...
        (3, 181, 217, 181) 138
        >>> with TiffFile('_vaa3d.v3draw.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(v3draw|raw)$'
    _filesizemin = 24 + 1 + 2 + 4 * 2  # 2 byte format
    _figureargs = {'figsize': (8, 8)}

    def _init(self, **kwargs: Any) -> None:
        """Read header and validate file size."""
        assert self._fh is not None
        # read 2 byte format header
        header = self._fh.read(24 + 1 + 2 + 4 * 2)
        # first 24 bytes are 'raw_image_stack_by_hpeng'
        if not header.startswith(b'raw_image_stack_by_hpeng'):
            raise LfdFileError(self)
        # next byte is byte order
        byteorder = {b'B': '>', b'L': '<'}[header[24:25]]
        # next two bytes are data itemsize and dtype
        itemsize = struct.unpack(byteorder + 'h', header[25:27])[0]
        self.dtype = numpy.dtype(
            byteorder + {1: 'u1', 2: 'u2', 4: 'f4'}[itemsize]
        )
        # next 8 or 16 bytes are data shape
        self.shape = struct.unpack(byteorder + 'hhhh', header[27:])[::-1]
        if self._filesize != len(header) + product(self.shape) * itemsize:
            # 4 byte format
            header += self._fh.read(8)
            self.shape = struct.unpack(byteorder + 'IIII', header[27:])[::-1]
            if self._filesize != len(header) + product(self.shape) * itemsize:
                raise LfdFileError(self, 'file size mismatch')
        self.axes = 'CZYX'

    def _asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Return data as array."""
        assert self._fh is not None
        assert self.dtype is not None
        assert self.shape is not None
        data = numpy.fromfile(self._fh, self.dtype)
        return data.reshape(*self.shape)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in file."""
        data = self._asarray()
        imshow(
            data, figure=figure, title=self._filename, photometric='MINISBLACK'
        )

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        tif.write(self.asarray(), **kwargs)


def vaa3draw_write(
    filename: os.PathLike[Any] | str,
    data: ArrayLike,
    /,
    *,
    byteorder: Literal['>', '<'] | None = None,
    twobytes: bool = False,
) -> None:
    """Write data to Vaa3D RAW binary file(s).

    Refer to :py:class:`Vaa3dRaw` for the v3draw file format.

    Parameters:
        filename:
            Name of file to write.
        data:
            Data to write. Must be of type uint8, uint16, or float32 with
            up to 5 dimensions ordered 'TCZYX'.
            Time points are stored in separate files.
        byteorder:
            Byte order of data in file.
        twobytes:
            If True, store data shape as int16, else uint32 (default).

    Examples:
        >>> data = (
        ...     numpy.arange(1000000)
        ...     .reshape(10, 10, 100, 100)
        ...     .astype('uint16')
        ... )
        >>> vaa3draw_write('_test.v3draw', data, byteorder='<')
        >>> with Vaa3dRaw('_test.v3draw') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """
    data = numpy.array(data, order='C', ndmin=5, copy=False)
    if data.dtype.char not in 'BHf':
        raise ValueError(f'invalid data type {data.dtype}')
    if data.ndim != 5:
        raise ValueError('data must be up to 5 dimensional')
    if byteorder is None:
        byteorder = '<' if sys.byteorder == 'little' else '>'
    elif byteorder not in {'>', '<'}:
        raise ValueError(f'invalid byteorder {byteorder}')
    assert byteorder is not None  # for mypy
    itemsize = {'B': 1, 'H': 2, 'f': 4}[data.dtype.char]
    dtype = byteorder + {1: 'u1', 2: 'u2', 4: 'f4'}[itemsize]
    header = b'raw_image_stack_by_hpeng'
    header += {'<': b'L', '>': b'B'}[byteorder]
    header += struct.pack(
        byteorder + ['hIIII', 'hhhhh'][bool(twobytes)],
        itemsize,
        *data.shape[:0:-1],
    )
    if data.shape[0] > 1:
        fmt = '%%s.t{t:0%i}%%s' % int(math.log(data.shape[0], 10) + 1)
        filenamefmt = fmt % os.path.splitext(filename)
    else:
        filenamefmt = str(filename)
    for t in range(data.shape[0]):
        with open(filenamefmt.format(t=t), 'wb') as fh:
            fh.write(header)
            data[t].astype(dtype, copy=False).tofile(fh)


class VoxxMap(LfdFile):
    """Voxx color palette.

    Voxx map files contain a single RGB color palette, stored as 256x4
    whitespace separated integers (0..255) in an ASCII file.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with VoxxMap('voxx.map') as f:
        ...     data = f.asarray()
        ...     f.totiff('_voxx.map.tif')
        ...     print(f.shape, data[100])
        ...
        (256, 3) [255 227 155 237]
        >>> with TiffFile('_voxx.map.tif') as f:
        ...     assert_array_equal(f.asarray()[0], data)
        ...

    """

    _filemode = 'r'
    _fileencoding = 'latin-1'
    _filepattern = r'.*\.map$'
    _figureargs = {'figsize': (6, 1)}

    def _init(self, **kwargs: Any) -> None:
        """Verify file starts with numbers."""
        assert self._fh is not None
        try:
            for i in self._fh.read(32).strip().split():
                if 0 <= int(i) <= 255:
                    continue
                raise ValueError('number out of range')
        except Exception as exc:
            raise LfdFileError(self) from exc
        self.shape = (256, 3)
        self.dtype = numpy.dtype(numpy.uint8)
        self.axes = 'XS'

    def _asarray(self, **kwargs: Any) -> NDArray[numpy.uint8]:
        """Return palette data as uint8 array of shape (256, 3)."""
        assert (
            self._fh is not None
            and self.shape is not None
            and self.dtype is not None
        )
        assert self._fh is not None
        self._fh.seek(0)
        data = numpy.fromfile(self._fh, numpy.uint8, 1024, sep=' ')
        return data.reshape(256, 4)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display palette stored in file."""
        pal = self.asarray().reshape(1, 256, -1)
        ax = figure.add_subplot(1, 1, 1)
        ax.set_title(self._filename)
        ax.yaxis.set_visible(False)
        ax.imshow(pal, aspect=20, origin='lower', interpolation='nearest')

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write palette to TIFF file."""
        kwargs.update(photometric='rgb', planarconfig='contig')
        data = numpy.expand_dims(self.asarray(), axis=0)
        tif.write(data, **kwargs)


def voxxmap_write(
    filename: os.PathLike[Any] | str,
    data: ArrayLike,
    /,
) -> None:
    """Write data to Voxx map file(s).

    Refer to :py:class:`VoxxMap` for the Voxx map format.

    Parameters:
        filename:
            Name of file to write.
        data:
            Data to write. Must be of shape (256, 4) and type uint8.

    Examples:
        >>> data = numpy.repeat(numpy.arange(256, dtype='uint8'), 4).reshape(
        ...     -1, 4
        ... )
        >>> voxxmap_write('_test_vox.map', data)
        >>> with VoxxMap('_test_vox.map') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """
    data = numpy.array(data, copy=False)
    if data.dtype.char != 'B' or data.shape != (256, 4):
        raise ValueError('not a 256x4 uint8 array')
    numpy.savetxt(filename, data, fmt='%.0f')


class NetpbmFile(LfdFile):
    """Netpbm formatted files.

    Netpbm files contain image data in a variety of formats as specified
    at http://netpbm.sourceforge.net/doc/.

    The following Netpbm and Portable FloatMap formats are supported:
    PBM (bi-level), PGM (grayscale), PPM (color), PAM (arbitrary),
    XV thumbnail (RGB332), PF (float32 RGB), and Pf (float32 grayscale).

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with NetpbmFile('netpbm.pam') as f:
        ...     data = f.asarray()
        ...     f.totiff('_netpbm.pam.tif')
        ...     print(f.shape, data[75, 75, 1])
        ...
        (150, 150, 4) 255
        >>> with TiffFile('_netpbm.pam.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(pnm|pbm|pgm|ppm|pam|pfm|xv)$'
    _figureargs = {'figsize': (8, 6)}

    # _netpbm: netpbmfile.NetpbmFile

    def _init(self, **kwargs: Any) -> None:
        """Validate file is a Netpbm file."""
        assert self._fh is not None
        if self._fh.read(2) not in {
            b'P1',
            b'P2',
            b'P3',
            b'P4',
            b'P5',
            b'P6',
            b'P7',
            b'PF',
            b'Pf',
        }:
            raise LfdFileError(self)
        self._fh.seek(0)

        import netpbmfile

        try:
            self._netpbm = netpbmfile.NetpbmFile(self._fh, **kwargs)
        except Exception as exc:
            raise LfdFileError(self) from exc

        self.shape = self._netpbm.shape
        self.dtype = self._netpbm.dtype
        self.axes = self._netpbm.axes

    def _asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Return data from Netpbm file."""
        return self._netpbm.asarray(**kwargs)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in file."""
        imshow(self.asarray(**kwargs), figure=figure, title=self._filename)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        kwargs.update(metadata=None)
        data = self.asarray()
        if data.ndim > 2 and data.shape[-1] in {3, 4}:
            kwargs.update(photometric='rgb', planarconfig='contig')
        tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return info about Netpbm file as string."""
        # return str(self._netpbm)

    def __getattr__(self, name: str, /) -> Any:
        """Return attribute from underlying NetpbmFile object."""
        return getattr(self._netpbm, name)


class OifFile(LfdFile):
    """Olympus(r) Image Format files (OIF and OIB).

    Olympus Image Format is the native file format of the Olympus
    FluoView(tm) software for confocal microscopy.

    This class is a light wrapper of the `oiffile` module. Use the module
    directly to access metadata in OIF and OIB files.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with OifFile('oiffile.oib') as f:
        ...     print(f.asarray()[2, 100, 100])
        ...
        248

    """

    _filepattern = r'.*\.(oib|oif)$'
    _figureargs = {'figsize': (8, 7)}

    def _init(self, **kwargs: Any) -> None:
        """Open OIF or OIB file."""
        assert self._fh is not None
        start = self._fh.read(8)
        if (
            start != b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
            and start[:4] != b'\xff\xfe\x5b\x00'
        ):
            raise LfdFileError(self)

        import oiffile

        self._oif: oiffile.OifFile | None
        try:
            self._oif = oiffile.OifFile(self.filename, **kwargs)
        except Exception as exc:
            raise LfdFileError(self) from exc

        self.axes = self._oif.axes
        self.shape = self._oif.shape
        self.dtype = self._oif.dtype

    def _close(self) -> None:
        """Close OifFile instance."""
        if self._oif is not None:
            self._oif.close()
            self._oif = None

    def _asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Return data from OIF file."""
        assert self._oif is not None
        return self._oif.asarray(**kwargs)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in file."""
        imshow(self.asarray(**kwargs), figure=figure, title=self._filename)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        data = self.asarray()
        if data.ndim > 2 and data.shape[-1] in {3, 4}:
            kwargs.update(photometric='rgb', planarconfig='contig')
        tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return info about OIF as string."""
        # return str(self._oif).split('\n', 2)[-1]

    def __getattr__(self, name: str, /) -> Any:
        """Return attribute from underlying OifFile object."""
        return getattr(self._oif, name)


class CziFile(LfdFile):
    """CZI file.

    Carl Zeiss Image (CZI) is the native file format of the ZEN(r) software
    by Carl Zeiss(r) Microscopy GmbH.

    This class is a light wrapper of the `czifile` module. Use the module
    directly to access additional image series and metadata in CZI files.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> with CziFile('czifile.czi') as f:
        ...     data = f.asarray()
        ...     f.totiff('_czifile.czi.tif')
        ...     print(f.shape, data[2, 2, 2, 80, 32, 2])
        ...
        (3, 3, 3, 250, 200, 3) 255
        >>> with TiffFile('_czifile.czi.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(czi)$'
    _figureargs = {'figsize': (8, 6)}

    def _init(self, **kwargs: Any) -> None:
        """Validate file is a CZI file."""
        assert self._fh is not None
        if self._fh.read(10) != b'ZISRAWFILE':
            raise LfdFileError(self)
        self._fh.seek(0)

        import czifile

        self._czi: czifile.CziFile | None
        try:
            self._czi = czifile.CziFile(self._fh, **kwargs)
        except Exception as exc:
            raise LfdFileError(self) from exc

        self.axes = self._czi.axes
        self.shape = self._czi.shape
        self.dtype = self._czi.dtype

    def _close(self) -> None:
        """Close OifFile instance."""
        if self._czi is not None:
            self._czi.close()
            self._czi = None

    def _asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Return data from CZI file."""
        assert self._czi is not None
        return self._czi.asarray(**kwargs)  # type: ignore[no-any-return]

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in file."""
        imshow(self.asarray(**kwargs), figure=figure, title=self._filename)

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        data = self.asarray()
        if data.ndim > 2 and data.shape[-1] <= 4:
            kwargs.update(planarconfig='contig')
        tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return CZI info as string."""
        # return str(self._czi).split('\n', 2)[-1]

    def __getattr__(self, name: str, /) -> Any:
        """Return attribute from underlying CziFile object."""
        return getattr(self._czi, name)


class TiffFile(LfdFile):
    """TIFF file.

    TIFF, the Tagged Image File Format, is used to store image and meta data
    from microscopy. Many custom extensions to the standard exist, such as
    LSM, STK, FluoView, MicroManager, ImageJ and OME-TIFF.

    This class is a light wrapper of the `tifffile` module. Use the module
    directly to access additional image series and metadata in TIFF files.

    Parameters:
        series: Select image series in TIFF file to read.

    Examples:
        >>> with TiffFile('tifffile.tif') as f:
        ...     data = f.asarray()
        ...     f.totiff('_tifffile.tif.tif')
        ...     print(f.shape, data[31, 30, 2])
        ...
        (32, 31, 3) 80
        >>> with TiffFile('_tifffile.tif.tif') as f:
        ...     assert_array_equal(f.asarray(), data)
        ...

    """

    _filepattern = r'.*\.(tif|tiff|stk|lsm|tf8)$'
    _figureargs = {'figsize': (8, 6)}

    _tif: tifffile.TiffFile | None
    _series: tifffile.TiffPageSeries

    def _init(self, *, series: int = 0, **kwargs: Any) -> None:
        """Validate file is a TIFF file."""
        assert self._fh is not None
        if self._fh.read(4) not in {
            b'MM\x00*',
            b'II*\x00',
            b'MM\x00+',
            b'II+\x00',
        }:
            raise LfdFileError(self)
        self._fh.seek(0)

        try:
            self._tif = tifffile.TiffFile(self._fh, **kwargs)
        except Exception as exc:
            raise LfdFileError(self) from exc

        self._series = self._tif.series[series]
        self.axes = self._series.axes
        self.shape = self._series.shape
        self.dtype = self._series.dtype

    def _close(self) -> None:
        """Close OifFile instance."""
        if self._tif is not None:
            self._tif.close()
            self._tif = None

    def _asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Return data from TIFF file."""
        assert self._tif is not None
        return self._tif.asarray(**kwargs)

    def _plot(self, figure: Figure, /, **kwargs: Any) -> None:
        """Display images stored in file."""
        page = self._series.keyframe
        imshow(
            self.asarray(**kwargs),
            figure=figure,
            title=self._filename,
            photometric=page.photometric,
            bitspersample=page.bitspersample,
        )

    def _totiff(self, tif: TiffWriter, /, **kwargs: Any) -> None:
        """Write image data to TIFF file."""
        data = self.asarray()
        if data.ndim > 2 and data.shape[-1] in {3, 4}:
            kwargs.update(photometric='rgb', planarconfig='contig')
        tif.write(data, **kwargs)

    def _str(self) -> str | None:
        """Return TIFF info as string."""
        # return str(self._tif)

    def __getattr__(self, name: str, /) -> Any:
        """Return attribute from underlying TiffFile object."""
        return getattr(self._tif, name)


def convert_fbd2b64(
    fbdfile: str | os.PathLike[Any],
    /,
    b64files: str = '{filename}_c{channel:02}t{frame:04}.b64',
    *,
    show: bool = True,
    verbose: bool = True,
    integrate_frames: int = 0,
    square_frame: bool = True,
    pdiv: int = -1,
    laser_frequency: float = -1,
    laser_factor: float = -1.0,
    pixel_dwell_time: float = -1.0,
    frame_size: int = -1,
    scanner_line_length: int = -1,
    scanner_line_start: int = -1,
    scanner_frame_start: int = -1,
    cmap: str = 'turbo',
) -> None:
    """Convert SimFCS FLIMbox data file to B64 files.

    Parameters:
        fbdfile:
            FLIMbox data file to convert.
        b64files:
            Format string for B64 output file names.
        show:
            If True, plot the decoded frames.
            If False, one B64 files is written for each frame and channel.
        verbose:
            Print detailed information about file and conversion.
        integrate_frames:
            Specifies which frames to sum. By default, no frames are summed.
        square_frame:
            If True, return square image (frame_size x frame_size).
            Else return full scanner frame.
        pdiv:
            Divisor to reduce number of entries in phase histogram.
        laser_frequency:
            Laser frequency in Hz.
        laser_factor:
            Factor to correct dwell_time/laser_frequency.
        pixel_dwell_time:
            Number of microseconds the scanner remains at each pixel.
        frame_size:
            Number of pixels in one line scan, excluding retrace.
        scanner_line_length:
            Number of pixels in each line, including retrace.
        scanner_line_start:
            Index of first valid pixel in scan line.
        scanner_frame_start:
            Index of first valid pixel after marker.
        cmap:
            Matplotlib colormap for plotting images.

    """
    prints: Any = print if verbose else nullfunc

    with FlimboxFbd(
        fbdfile,
        pdiv=pdiv,
        laser_frequency=laser_frequency,
        laser_factor=laser_factor,
        pixel_dwell_time=pixel_dwell_time,
        frame_size=frame_size,
        scanner_line_length=scanner_line_length,
        scanner_line_start=scanner_line_start,
        scanner_frame_start=scanner_frame_start,
    ) as fbd:
        prints(fbd)
        decoded = fbd.decode()
        bins, times, markers = decoded
        # prints(decoded)
        frames = fbd.frames(decoded)
        # prints(frames)
        image = fbd.asimage(
            decoded,
            frames,
            integrate_frames=integrate_frames,
            square_frame=square_frame,
        )
        # prints(image)
        image = numpy.moveaxis(image, -1, 2)
        times_ = times / (fbd.laser_frequency * fbd.laser_factor)
        if show:
            from matplotlib import pyplot
            from tifffile import imshow

            pyplot.figure()
            pyplot.title('Bins and Markers')
            pyplot.xlabel('time (s)')
            pyplot.ylabel('Cross correlation phase index')
            pyplot.plot(times_, bins[0], '.', alpha=0.5, label='Ch0')
            pyplot.vlines(
                x=times_[markers],
                ymin=-1,
                ymax=1,
                colors='red',
                label='Markers',
            )
            pyplot.legend()

            pyplot.figure()
            pyplot.title('Cross correlation phase histogram')
            pyplot.xlabel('Bin')
            pyplot.ylabel('Counts')
            bins_channel: Any  # for mypy
            for ch, bins_channel in enumerate(decoded[0]):
                histogram = numpy.bincount(bins_channel[bins_channel >= 0])
                pyplot.plot(histogram, label=f'Ch{ch}')
            pyplot.legend()

            if not integrate_frames and image.shape[0] > 1:
                image = numpy.moveaxis(image, 0, 1)
                title = 'Photons[Channel, Frame, Bin, Y, X]'
            else:
                title = 'Photons[Channel, Bin, Y, X]'
            imshow(
                image,
                vmax=None,
                photometric='minisblack',
                cmap=cmap,
                title=title,
            )
            pyplot.show()
        else:
            prints()
            for j in range(image.shape[0]):
                for i in range(image.shape[1]):
                    b64name = b64files.format(
                        filename=fbdfile, channel=i, frame=j
                    )
                    prints(b64name, flush=True)
                    simfcsb64_write(
                        b64name, image[j, i].astype(numpy.int16, copy=False)
                    )


def convert2tiff(
    files: str | os.PathLike[Any] | Sequence[str | os.PathLike[Any]],
    /,
    *,
    verbose: bool = True,
    skip: Sequence[type[LfdFile]] | None = None,
    **kwargs: Any,
) -> None:
    """Convert image data from LfdFile(s) to TIFF files.

    Parameters:
        files:
            Files to convert to TIFF.
        verbose:
            Print conversion status.
        skip:
            File types not to use for conversion.
            The default is SimfcsBin, SimfcsRaw, SimfcsCyl, FliezI16.

    Examples:
        >>> convert2tiff('flimfast.flif')
        flimfast.flif - FlimfastFlif

    """
    if skip is None:
        skip = SimfcsBin, SimfcsRaw, SimfcsCyl, FliezI16

    registry = [
        cls
        for cls in LfdFileRegistry.classes
        if cls not in skip and cls._totiff != LfdFile._totiff
    ]
    for file in LfdFileSequence(files, imread=LfdFile):
        if verbose:
            print(file, end=' - ')
            sys.stdout.flush()
        for cls in registry:
            try:
                with cls(file, validate=True) as fh:  # typing: ignore
                    fh.totiff(**kwargs)
                if verbose:
                    print(cls.__name__)
                break
            except LfdFileError:
                pass
            except Exception as exc:
                if verbose:
                    print(exc, end=' - ')
        else:
            if verbose:
                print('failed')
            continue
        registry.remove(cls)
        registry.insert(0, cls)


def uleb128(leb: bytes, /) -> tuple[int, int]:
    """Return little-endian base 128 decoded integer."""
    i = 0
    shift = 0
    value = 0
    for b in leb:
        i += 1
        value += (b & 0x7F) << shift
        if b >> 7 == 0:
            break
        shift += 7
    return value, i


def read_record(
    meta: dict[str, Any],
    fields: dict[str, str],
    fh: Any,
    /,
    *,
    byteorder: Literal['>', '<'] = '<',
) -> int:
    """Read record from file and add fields to meta."""
    size = 0
    for fmt in fields.values():
        size += struct.calcsize(fmt)
    data = fh.read(size)
    offset = 0
    for name, fmt in fields.items():
        value = struct.unpack_from(byteorder + fmt, data, offset)
        if not fmt[0].isnumeric() and len(value) == 1:
            value = value[0]
        meta[name] = value
        offset += struct.calcsize(fmt)
    return size


def determine_shape(
    shape: tuple[int, ...],
    dtype: DTypeLike,
    size: int,
    /,
    *,
    validate: bool = True,
    exception: type[Exception] = LfdFileError,
) -> tuple[int, ...]:
    """Validate and return array shape from dtype and data size.

    Parameters:
        shape:
            Shape of array. One shape dimension can be -1. In this case,
            the value is inferred from size and remaining dimensions.
        dtype:
            Datatype of array.
        size:
            Size of array data in bytes.
        validate:
            If True, 'size' must exactly match 'shape' and 'dtype'.

    Examples:
        >>> determine_shape((-1, 2, 2), 'uint16', 16)
        (2, 2, 2)

    """
    dtype = numpy.dtype(dtype)
    undetermined = len([i for i in shape if i < 0])
    if undetermined > 1:
        raise ValueError('invalid shape')
    if size < 0:
        raise ValueError('invalid size')
    if undetermined:
        count = int(size // dtype.itemsize)
    else:
        count = product(shape)
        if count * dtype.itemsize > size:
            raise exception('file is too small')
    if validate and count * dtype.itemsize != size:
        raise exception('file size mismatch')
    if undetermined:
        t = count // product(i for i in shape if i > 0)
        shape = tuple((i if i > 0 else t) for i in shape)
    return shape


def indent(*args: str) -> str:
    """Return joined string representations of objects with indented lines."""
    text = '\n'.join(str(arg) for arg in args)
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


def format_dict(
    adict: dict[str, Any],
    /,
    *,
    prefix: str = '',
    indent: str = '  ',
    bullets: tuple[str, str] = ('', ''),
    excludes: Sequence[str] = ('_',),
    linelen: int = 79,
    trim: int = 0,
) -> str:
    """Return pretty-print of nested dictionary."""
    result = []
    for k, v in sorted(adict.items(), key=lambda x: str(x[0]).lower()):
        if any(k.startswith(e) for e in excludes):
            continue
        if isinstance(v, dict):
            v = '\n' + format_dict(
                v, prefix=prefix + indent, excludes=excludes, trim=0
            )
            result.append(f'{prefix}{bullets[1]}{k}: {v}')
        else:
            result.append((f'{prefix}{bullets[0]}{k}: {v}')[:linelen].rstrip())
    if trim > 0:
        result[0] = result[0][trim:]
    return '\n'.join(result)


def stripnull(string: bytes) -> bytes:
    r"""Return byte string truncated at first null character.

    Use to clean NULL terminated C strings.

    >>> stripnull(b'bytes\x00\x00b')
    b'bytes'

    """
    i = string.find(b'\x00')
    return string if i < 0 else string[:i]


def bytes2str(
    b: bytes, /, encoding: str | None = None, errors: str = 'strict'
) -> str:
    """Return Unicode string from encoded bytes up to first NULL character."""
    if encoding is None or '16' not in encoding:
        i = b.find(b'\x00')
        if i >= 0:
            b = b[:i]
    else:
        # utf-16
        i = b.find(b'\x00\x00')
        if i >= 0:
            b = b[: i + i % 2]

    try:
        return b.decode('utf-8' if encoding is None else encoding, errors)
    except UnicodeDecodeError:
        if encoding is not None:
            raise
        return b.decode('cp1252', errors)


def nullfunc(*args: Any, **kwargs: Any) -> None:
    """Null function."""
    return


def logger() -> logging.Logger:
    """Return logger for lfdfiles module."""
    return logging.getLogger('lfdfiles')


def main() -> None:
    """Command line usage main function."""
    import click

    @click.group()
    @click.version_option(version=__version__)
    def cli() -> None:
        pass

    @cli.command(help='Convert files to TIFF.')
    @click.option(
        '--format',
        default='tiff',
        help='Output file format.',
        type=click.Choice(['tiff']),
    )
    @click.option(
        '--compress',
        default=0,
        help='Zlib compression level.',
        type=click.IntRange(0, 10, clamp=False),
    )
    @click.argument('files', nargs=-1, type=click.Path(dir_okay=False))
    def convert(format: str, compress: int, files: Any) -> None:
        if not files:
            files = askopenfilename(
                title='Select LFD file(s)',
                multiple=True,
                filetypes=[('All files', '*')],
            )
        if files:
            convert2tiff(files, compress=compress)

    @cli.command(help='View data in file.')
    @click.argument('files', nargs=-1, type=click.Path(dir_okay=False))
    def view(files: Any) -> None:
        if not files:
            files = askopenfilename(
                title='Select LFD file(s)', filetypes=[('All files', '*')]
            )
        if files:
            if isinstance(files, (list, tuple)):
                files = files[0]
            with LfdFile(files) as fh:
                print(fh)
                fh.show()

    if len(sys.argv) == 1:
        sys.argv.append('view')
    elif len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        sys.argv.append(sys.argv[1])
        sys.argv[1] = 'view'

    cli(prog_name='lfdfiles')


LfdFileRegistry.sort()

if __name__ == '__main__':
    main()
