# lfdfiles.py

# Copyright (c) 2012-2021, Christoph Gohlke
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
converting, and viewing many of the proprietary file formats used to store
experimental data and metadata at the `Laboratory for Fluorescence Dynamics
<https://www.lfd.uci.edu/>`_. For example:

* SimFCS VPL, VPP, JRN, BIN, INT, CYL REF, BH, BHZ FBF, FBD, B64, I64, Z64, R64
* GLOBALS LIF, ASCII
* CCP4 MAP
* Vaa3D RAW
* Bio-Rad(r) PIC
* Vista IFLI
* FlimFast FLIF

For command line usage run ``python -m lfdfiles --help``

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2021.6.6

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.7.9, 3.8.10, 3.9.5 64-bit <https://www.python.org>`_
* `Cython 0.29.23 <https://cython.org>`_ (build)
* `Numpy 1.20.3 <https://pypi.org/project/numpy/>`_
* `Tifffile 2021.4.8 <https://pypi.org/project/tifffile/>`_  (optional)
* `Czifile 2019.7.2 <https://pypi.org/project/czifile/>`_ (optional)
* `Oiffile 2021.6.6 <https://pypi.org/project/oiffile />`_ (optional)
* `Netpbmfile 2021.6.6 <https://pypi.org/project/netpbmfile />`_ (optional)
* `Matplotlib 3.4.2 <https://pypi.org/project/matplotlib/>`_
  (optional for plotting)
* `Click 8.0 <https://pypi.python.org/pypi/click>`_
  (optional for command line usage)

Revisions
---------
2021.6.6
    Fix unclosed file warnings.
    Replace TIFF compress with compression parameter (breaking).
    Remove compress option from command line interface (breaking).
2021.2.22
    Add function to decode Spectral FLIM data from Kintex FLIMbox.
    Relax VistaIfli file version check.
2020.9.18
    Remove support for Python 3.6 (NEP 29).
    Support os.PathLike file names.
    Fix writing contiguous series to TIFF files with tifffile >= 2020.9.3.
2020.1.1
    Read CZI files via czifile module.
    Read Olympus Image files via oiffile module.
    Read Netpbm formats via netpbmfile module.
    Add B64, Z64, and I64 write functions.
    Remove support for Python 2.7 and 3.5.
    Update copyright.
2019.7.2
   Require tifffile 2019.7.2.
   Remove some utility functions.
2019.5.22
    Read and write Bio-Rad(tm) PIC files.
    Read and write Voxx MAP palette files.
    Rename SimfcsMap to Ccp4Map and SimfcsV3draw to Vaa3dRaw.
    Rename save functions.
2019.4.22
    Fix setup requirements.
2019.1.24
    Add plots for GlobalsLif, SimfcsV3draw, and VistaIfli.
    Support Python 3.7 and numpy 1.15.
    Move modules into lfdfiles package.
2018.5.21
    Update SimfcsB64 to handle carpets and streams.
    Command line interface for plotting and converting to TIFF.
    Registry of LfdFile classes.
    Write image and metadata to TIFF.
    Read TIFF files via tifffile module.
2016.3.29
    Add R64 write function.
2016.3.14
    Read and write Vaa3D RAW volume files.
2015.3.02
    Initial support for plotting.
2015.2.19
    Initial support for new FBD files containing headers.
2014.12.2
    Read B64, R64, I64 and Z64 files (SimFCS version 4).
2014.10.10
    Read SimFCS FIT files.
2014.4.8
    Read and write CCP4 MAP volume files.
2013.8.10
    Read second harmonics FlimBox data.

Notes
-----
Lfdfiles is currently developed, built, and tested on Windows only.

The API is not stable yet and might change between revisions.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017
and 2019 <https://support.microsoft.com/en-us/help/2977003/
the-latest-supported-visual-c-downloads>`_ is required on Windows.

Many of the LFD's file formats are not documented and might change arbitrarily.
This implementation is mostly based on reverse engineering existing files.
No guarantee can be made as to the correctness of code and documentation.

Experimental data are often stored in plain binary files with metadata
available in separate, human readable journal files (.jrn).

Unless specified otherwise, data are stored in little-endian, C contiguous
order.

Examples
--------
Create a Bio-Rad PIC file from a numpy array:

>>> data = numpy.arange(1000000).reshape(100, 100, 100).astype('u1')
>>> bioradpic_write('_biorad.pic', data)

Read the volume data from the PIC file as numpy array, and access metadata:

>>> with BioradPic('_biorad.pic') as f:
...     f.shape
...     f.spacing
...     data = f.asarray()
(100, 100, 100)
(1.0, 1.0, 1.0)

Convert the PIC file to a compressed TIFF file:

>>> with BioradPic('_biorad.pic') as f:
...     f.totiff('_biorad.tif', compression='zlib')


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
4.  `FlimFast <https://www.lfd.uci.edu/~gohlke/flimfast/>`_ is software for
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
11. `Voxx <http://www.indiana.edu/~voxx/>`_ is a volume rendering program
    for 3D microscopy, developed by Jeff Clendenon et al. at the Indiana
    University.
12. `CCP4 <https://www.ccp4.ac.uk/>`_, the Collaborative Computational Project
    No. 4, is software for macromolecular X-Ray crystallography.

"""

__version__ = '2021.6.6'

__all__ = (
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
    'SimfcsFbf',
    'SimfcsFbd',
    'SimfcsGpSeries',
    'SimfcsB64',
    'SimfcsI64',
    'SimfcsZ64',
    'SimfcsR64',
    'GlobalsLif',
    'GlobalsAscii',
    'VistaIfli',
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
    'convert2tiff',
    'simfcsb64_write',
    'simfcsi64_write',
    'simfcsz64_write',
    'simfcsr64_write',
    'ccp4map_write',
    'vaa3draw_write',
    'voxxmap_write',
    'bioradpic_write',
    'simfcsfbd_histogram',
    'simfcsfbd_decode',
    'sflim_decode',
)

import os
import sys
import re
import math
import struct
import zlib
import zipfile
import warnings

import numpy

import tifffile
from tifffile import (
    TiffWriter,
    FileSequence,
    imshow,
    astype,
    product,
    stripnull,
    update_kwargs,
    parse_kwargs,
    askopenfilename,
)

# delay import optional modules
pyplot = None
cycler = None


def import_pyplot(fail=True):
    """Import matplotlib.pyplot."""
    global pyplot
    global cycler

    if pyplot is not None:
        return True
    try:
        from matplotlib import pyplot as plt
        import cycler as cyclr

        pyplot = plt
        cycler = cyclr
    except ImportError:
        if fail:
            raise
        return False
    return True


class lazyattr:
    """Lazy object attribute whose value is computed on first access."""

    __slots__ = ('func',)

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        result = self.func(instance)
        if result is NotImplemented:
            return getattr(super(owner, instance), self.func.__name__)
        setattr(instance, self.func.__name__, result)
        return result


class LfdFileError(Exception):
    """Exception to indicate invalid LFD files."""

    def __init__(self, arg='', msg=''):
        if isinstance(arg, LfdFile):
            arg = f'not a valid {arg.__class__.__name__} file'
        if msg:
            arg = f'{arg}: {msg}' if arg else msg
        Exception.__init__(self, arg)


class LfdFileRegistry(type):
    """Metaclass to register classes derived from LfdFile."""

    classes = []

    def __new__(cls, name, bases, dct):
        klass = type.__new__(cls, name, bases, dct)
        if klass.__name__[:7] != 'LfdFile':
            LfdFileRegistry.classes.append(klass)
        return klass


LfdFileBase = LfdFileRegistry('LfdFileBase', (object,), {})


class LfdFile(LfdFileBase):
    """Base class for reading LFD files.

    Attributes
    ----------
    shape : tuple or None
        Shape of array data contained in file.
    dtype : numpy.dtype or None
        Type of array data contained in file.
    axes : str or None
        Character labels for axes of array data in file.

    Examples
    --------
    >>> with LfdFile('flimfast.flif') as f:
    ...     type(f)
    <class '...FlimfastFlif'>
    >>> with LfdFile('simfcs.ref', validate=False) as f:
    ...     type(f)
    <class '...SimfcsRef'>
    >>> with LfdFile('simfcs.bin', shape=(-1, 256, 256), dtype='u2') as f:
    ...     type(f)
    <class '...SimfcsBin'>

    """

    _filemode = 'rb'  # file open mode
    _filepattern = r'.*'  # regular expression matching valid file names
    _filesizemin = 16  # minimum file size
    _figureargs = {'figsize': (6, 8.5)}  # arguments passed to pyplot.figure

    def __new__(cls, filename, *args, **kwargs):
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
                skip.update((SimfcsBh, SimfcsCyl, SimfcsFbd, FliezI16))
        for lfdfile in registry:
            if lfdfile in skip:
                continue
            try:
                with lfdfile(filename, *args, **kwargs):
                    pass
            except FileNotFoundError:
                raise
            except Exception:
                continue
            else:
                return super().__new__(lfdfile)
        raise LfdFileError('failed to read file using any LfdFile class')

    def __init__(self, filename, *args, **kwargs):
        """Open file(s) and read headers and metadata.

        Parameters
        ----------
        filename: str or path-like
            Name of file to open.
        validate : bool
            If True, filename must match the _filepattern regular expression.
        components : bool
            If True, open all component files found.

        """
        kwargs2 = parse_kwargs(kwargs, validate=True, components=True)
        components = kwargs2['components']
        validate = kwargs2['validate']
        self.shape = None
        self.dtype = None
        self.axes = None
        self._pos = None

        self._filepath, self._filename = os.path.split(os.fspath(filename))
        self.components = self._components() if components else []
        if validate:
            self._valid_name()
        if self.components:
            self._fh = None
            # verify file name is a component
            for label, fname in self.components:
                if fname.lower() == self._filename.lower():
                    break
            else:
                raise LfdFileError(self, 'not a component file')
            # try to open file using all registered classes
            components = []
            for label, fname in self.components:
                fname = os.path.join(self._filepath, fname)
                try:
                    lfdfile = self.__class__(
                        fname, validate=validate, components=False
                    )
                except Exception:  # LfdFileError, FileNotFoundError
                    continue
                components.append((label, lfdfile))
            if not components:
                raise LfdFileError(self, 'no component files found')
            self.components = components
            self.shape = (len(components),) + lfdfile.shape
            self.dtype = lfdfile.dtype
            if components[0][1].axes is not None:
                self.axes = 'S' + components[0][1].axes
        else:
            self._fh = open(filename, self._filemode)
            try:
                if self._filesizemin != len(self._fh.read(self._filesizemin)):
                    raise LfdFileError(self, 'file is too small')
            except LfdFileError:
                self._fh.close()
                self._fh = None
                raise
            except Exception:
                self._fh.close()
                self._fh = None
                raise LfdFileError(self, 'not a text file')

            self._fh.seek(0)
            try:
                self._init(*args, **kwargs)
            except Exception:
                if self._fh:
                    self._fh.close()
                    self._fh = None
                raise
            if self._fh:
                self._pos = self._fh.tell()

    def __str__(self):
        """Return string with information about file."""
        s = [
            self.__class__.__name__,
            os.path.normpath(os.path.normcase(self.filename)),
        ]
        if self.components:
            components = ', '.join(i for i, c in self.components)
            s.append(f'components: {components}')
        if self.axes is not None:
            s.append(f'axes: {self.axes}')
        if self.shape is not None:
            shape = ', '.join(str(i) for i in self.shape)
            s.append(f'shape: {shape}')
        if self.dtype is not None:
            s.append(f'dtype: {numpy.dtype(self.dtype)}')
        _str = self._str()
        if _str:
            s.append(_str)
        return '\n '.join(s)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Close file handle."""
        try:
            self._close()
        except Exception:
            pass
        if self._fh:
            self._fh.close()
            self._fh = None
        for _, component in self.components:
            component.close()

    def asarray(self, *args, **kwargs):
        """Return data in file(s) as numpy array."""
        if self.components:
            data = [c.asarray(*args, **kwargs) for _, c in self.components]
            return numpy.array(data).squeeze()
        if self._fh:
            self._fh.seek(self._pos)
        return self._asarray(*args, **kwargs)

    def totiff(self, filename=None, **kwargs):
        """Write image(s) and metadata to TIFF file."""
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

    def show(self, **kwargs):
        """Display data in matplotlib figure."""
        if self._plot is None:
            return
        import_pyplot()
        figure = pyplot.figure(facecolor='w', **self._figureargs)
        try:
            figure.canvas.manager.window.title('LfdFiles - ' + self._filename)
        except Exception:
            pass
        self._plot(figure, **kwargs)
        pyplot.show()

    def _init(self, *args, **kwargs):
        """Override to validate file and read metadata."""

    def _close(self):
        """Override to free any allocated resources."""

    def _components(self):
        """Override to return possible names of component files."""
        return []

    def _asarray(self, *args, **kwargs):
        """Override to read data from file and return as numpy array."""
        raise NotImplementedError()

    def _totiff(self, tif, **kwargs):
        """Override to write images and metadata to TIFF file."""
        raise NotImplementedError()

    def _plot(self, figure, **kwargs):
        """Override to display data in matplotlib figure."""
        update_kwargs(kwargs, cmap='viridis')
        pyplot.subplots_adjust(bottom=0.07, top=0.93)
        try:
            data = self.asarray()
            if isinstance(data, tuple):
                data = data[0]
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
            bins = data.max() - data.min() if data.dtype.kind in 'iu' else 64
            bins = bins if bins else 64
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

    def _str(self):
        """Override to return extra information about file."""

    def _valid_name(self):
        """Raise LfdFileError if filename does not match _filepattern."""
        if not self._filepattern or self._filepattern == r'.*':
            return
        if re.search(self._filepattern, self._filename, re.IGNORECASE) is None:
            raise LfdFileError(
                self,
                ".\n    File name '{}' does not match '{}')".format(
                    self._filename, self._filepattern
                ),
            )

    def _decompress_header(self, max_length, max_read=256):
        """Return first uncompressed bytes of zlib compressed file."""
        data = self._fh.read(max_read)
        self._fh.seek(0)
        if not data.startswith(b'\x78\x9c'):
            raise LfdFileError(self, 'not a zlib compressed file')
        decompressor = zlib.decompressobj()
        return decompressor.decompress(data, max_length)

    @lazyattr
    def _fstat(self):
        """Return status of open file."""
        return os.fstat(self._fh.fileno())

    @lazyattr
    def _filesize(self):
        """Return file size in bytes."""
        pos = self._fh.tell()
        self._fh.seek(0, 2)
        size = self._fh.tell()
        self._fh.seek(pos)
        return size

    @property
    def filename(self):
        """Return name of file."""
        return os.path.join(self._filepath, self._filename)

    @property
    def size(self):
        """Return number of elements in data array."""
        if self.shape is None:
            return None
        return product(self.shape)

    @property
    def ndim(self):
        """Return number of dimensions in data array."""
        if self.shape is None:
            return None
        return len(self.shape)


class LfdFileSequence(FileSequence):
    """Series of LFD files.

    Attributes
    ----------
    files : list of str
        List of file names.
    shape : tuple
        Shape of image sequence detected from file names.

    Examples
    --------
    >>> ims = LfdFileSequence('gpint/v?001.int', imread=SimfcsInt)
    >>> ims.axes
    'I'
    >>> data = ims.asarray()
    >>> data.shape
    (2, 256, 256)
    >>> ims.close()

    """

    _readfunction = None
    _indexpattern = None

    def __init__(
        self, files=None, imread=None, pattern=None, container=None, sort=None
    ):
        """Initialize instance from multiple LFD files."""
        if pattern is None:
            pattern = self._indexpattern
        if imread is None:
            imread = self._readfunction
        if imread is None:
            raise ValueError('imread function not specified')
        super().__init__(
            imread, files, container=container, sort=sort, pattern=pattern
        )


class RawPal(LfdFile):
    """Raw color palette.

    PAL files contain a single RGB or RGBA color palette, stored as 256x3 or
    256x4 unsigned bytes in C or Fortran order, without any header.

    Examples
    --------
    >>> with RawPal('rgb.pal') as f:
    ...     print(f.asarray()[100])
    [ 16 255 239]
    >>> with RawPal('rgba.pal') as f:
    ...     print(f.asarray()[100])
    [219 253 187 255]
    >>> with RawPal('rrggbb.pal') as f:
    ...     print(f.asarray()[100])
    [182 114  91]
    >>> with RawPal('rrggbbaa.pal') as f:
    ...     print(f.asarray()[100])
    [182 114  91 170]
    >>> with RawPal('rrggbbaa.pal') as f:
    ...     print(f.asarray(order='F')[100])
    [182 114  91 170]

    """

    _filepattern = r'.*\.(pal|raw|bin|lut)$'
    _figureargs = {'figsize': (6, 1)}

    def _init(self):
        """Verify file size is 768 or 1024."""
        if self._filesize not in (768, 1024):
            raise LfdFileError(self)
        self.shape = 256, -1
        self.dtype = numpy.dtype('u1')
        self.axes = 'XS'

    def _asarray(self, order=None):
        """Return palette data as (256, 3) or (256, 4) shaped array of uint8.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Determines whether the data is stored in C (row-major) or
            Fortran (column-major) order. By default the order is
            determined based on size and sum of differences.

        """
        data = numpy.fromfile(self._fh, 'u1').reshape(256, -1)
        if order is None:
            a = data.astype('i4')
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

    def _plot(self, figure):
        """Display palette stored in file."""
        pal = self.asarray().reshape(1, 256, -1)
        ax = figure.add_subplot(1, 1, 1)
        ax.set_title(self._filename)
        ax.yaxis.set_visible(False)
        ax.imshow(pal, aspect=20, origin='lower', interpolation='nearest')

    def _totiff(self, tif, **kwargs):
        """Write palette to TIFF file."""
        kwargs.update(photometric='rgb', planarconfig='contig')
        kwargs2 = parse_kwargs(kwargs, 'order')
        data = self.asarray(**kwargs2)
        data = numpy.expand_dims(data, axis=0)
        tif.save(data, **kwargs)


class SimfcsVpl(LfdFile):
    """SimFCS or ImObj color palette.

    SimFCS VPL files contain a single RGB color palette, stored as 256x3
    unsigned bytes in C or Fortran order, preceded by a 22 or 24 bytes header.

    Attributes
    ----------
    name : str or None
        Name of the palette.

    Examples
    --------
    >>> with SimfcsVpl('simfcs.vpl') as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs.vpl.tif')
    ...     print(f.shape, data[100])
    (256, 3) [189 210 246]
    >>> with TiffFile('_simfcs.vpl.tif') as f:
    ...     assert_array_equal(f.asarray()[0], data)

    >>> with SimfcsVpl('imobj.vpl') as f:
    ...     data = f.asarray()
    ...     f.totiff('_imobj.vpl.tif')
    ...     print(f.shape, data[100])
    (256, 3) [  0 254  27]
    >>> with TiffFile('_imobj.vpl.tif') as f:
    ...     assert_array_equal(f.asarray()[0], data)

    """

    _filepattern = r'.*\.vpl$'
    _figureargs = {'figsize': (6, 1)}

    def _init(self):
        """Verify file size and header."""
        if self._filesize == 792:
            self._fh.seek(24)
            self.name = None
        elif self._filesize >= 790:
            if self._fh.read(7) != b'vimage:':
                raise LfdFileError(self)
            self.name = bytes2str(stripnull(self._fh.read(15)))
        else:
            raise LfdFileError(self)
        self.shape = 256, 3
        self.dtype = numpy.dtype('u1')
        self.axes = 'XS'

    def _asarray(self):
        """Return palette data as (256, 3) shaped array of uint8."""
        data = numpy.fromfile(self._fh, 'u1', 768)
        if self._filesize == 792:
            return data.reshape(256, 3)
        return data.reshape(3, 256).T

    def _plot(self, figure):
        """Display palette stored in file."""
        pal = self.asarray().reshape(1, 256, -1)
        ax = figure.add_subplot(1, 1, 1)
        ax.set_title(self.name if self.name else self._filename)
        ax.yaxis.set_visible(False)
        ax.imshow(pal, aspect=20, origin='lower', interpolation='nearest')

    def _totiff(self, tif, **kwargs):
        """Write palette to TIFF file."""
        kwargs.update(
            photometric='rgb', planarconfig='contig', description=self.name
        )
        data = numpy.expand_dims(self.asarray(), axis=0)
        tif.save(data, **kwargs)

    def _str(self):
        """Return name of palette."""
        if self.name:
            return f'name: {self.name}'
        return None


class SimfcsVpp(LfdFile):
    """SimFCS color palettes.

    SimFCS VPP files contain multiple BGRA color palettes, each stored as
    256x4 values of unsigned bytes preceded by a 24 byte Pascal string.

    Attributes
    ----------
    names : list of str
        Names of all palettes in file.

    Examples
    --------
    >>> with SimfcsVpp('simfcs.vpp') as f:
    ...     data = f.asarray('nice.vpl')
    ...     f.totiff('_simfcs.vpp.tif')
    ...     print(f.shape, data[100])
    (256, 4) [ 16 255 239 255]
    >>> with TiffFile('_simfcs.vpp.tif') as f:
    ...     assert_array_equal(f.asarray()[35, 0], data)

    """

    _filepattern = r'.*\.vpp$'
    _filesizemin = 24

    def _init(self):
        """Read list of palette names from file."""
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
        self.dtype = numpy.dtype('u1')
        self.axes = 'XS'
        self._figureargs = {'figsize': (6, len(self.names) / 6)}

    def _asarray(self, key=0, rgba=True):
        """Return palette data as (256, 4) shaped array of uint8.

        Parameters
        ----------
        key : int or str, optional
            The index or name of the palette to return.
        rgba : bool, optional
            Return RGBA palette if True (default), else BGRA.

        """
        try:
            self.names[key]  # noqa: validation
        except TypeError:
            key = self.names.index(key)
        self._fh.seek(key * 1048 + 24)
        data = numpy.fromfile(self._fh, 'u1', 1024).reshape(256, 4)
        if rgba:
            data[:, :3] = data[:, 2::-1]
        if numpy.all(data[:, 3] == 0):
            data[:, 3] = 255  # fix transparency
        return data

    def _plot(self, figure):
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

    def _totiff(self, tif, **kwargs):
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
            tif.save(data, **kwargs)

    def __len__(self):
        return len(self.names)

    def _str(self):
        """Return names of all palettes in file."""
        return 'names:\n  {}'.format('\n  '.join(self.names))


class SimfcsJrn(LfdFile):
    """SimFCS journal.

    SimFCS JRN files contain metadata for several measurements, stored as
    key, value pairs in an unstructured ASCII format. Records usually start
    with lines of 80 '*' characters. The files do not contain array data.

    The metadata can be accessed as a list of dictionaries.

    Examples
    --------
    >>> with SimfcsJrn('simfcs.jrn', lower=True) as f:
    ...     f[1]['paramters for tracking']['samplimg frequency']
    15625

    """

    _filemode = 'r'
    _filepattern = r'.*\.jrn$'

    # regular expressions of all keys found in journal files
    _keys = r"""
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

    _keys = '|'.join(
        i.strip().replace(' ', '[ ]') for i in _keys.splitlines() if i.strip()
    )
    _keys = re.compile(f'({_keys})')
    _skip = re.compile(r'\s*\(.*\)')  # ignore parenthesis in values
    _plot = None

    def _init(self, lower=False):
        """Read journal file and parse into list of dictionaries.

        Parameters
        ----------
        lower : bool
            Convert keys to lower case.

        """
        firstline = self._fh.readline()
        if not firstline.startswith('*' * 80) or firstline.startswith('roi'):
            raise LfdFileError(self)
        content = self._fh.read()
        self._filesize = len(firstline) + len(content)
        if not firstline.startswith('***'):
            content = firstline + '\n' + content
        self._lower = lower = (lambda x: x.lower()) if lower else (lambda x: x)
        self._records = []
        for record in content.split('*' * 80):
            recdict = {}
            record = record.split('COMMENTS', 1)
            if len(record) > 1:
                record, comments = record
            else:
                record, comments = record[0], ''
            recdict[lower('COMMENTS')] = comments.strip('* :;=\n\r\t')
            record = re.split(r'[*]{5}([\w\s]+)[*]{5}', record)
            for key, value in zip(record[1::2], record[2::2]):
                newdict = {}
                key = lower(key.strip('* :;=\n\r\t'))
                value = self._parse_journal(
                    value, self._keys, result=newdict, lower=lower
                )
                recdict[key] = newdict
            self._parse_journal(
                record[0], self._keys, result=recdict, lower=lower
            )
            self._records.append(recdict)
        self.close()

    def _asarray(self):
        """Raise ValueError."""
        raise ValueError('file does not contain array data')

    @staticmethod
    def _parse_journal(journal, repattern, result=None, lower=None):
        """Return dictionary of keys and values in journal string."""
        if result is None:
            result = {}
        if lower is None:

            def lower(x):
                return x

        keyval = re.split(repattern, journal, maxsplit=0)
        keyval = list(s.strip('* :;=\n\r\t') for s in keyval)
        val = [astype(re.sub(SimfcsJrn._skip, '', v)) for v in keyval[2:-1:2]]
        key = [lower(k) for k in keyval[1:-1:2]]
        result.update(zip(key, val))
        return result

    def _str(self):
        """Return string with information about file."""
        lower = self._lower
        comments = lower('COMMENTS')
        s = []
        for i, record in enumerate(self._records):
            s.extend(
                (
                    f'Record {i}',
                    format_dict(record, '  ', excludes=[comments, '_']),
                    f" {lower('COMMENTS')}\n   {record[comments]}",
                )
            )
        return '\n '.join(s)

    def __getitem__(self, key):
        """Return selected record."""
        return self._records[key]

    def __len__(self):
        """Return number of records."""
        return len(self._records)

    def __iter__(self):
        """Return iterator over records."""
        return iter(self._records)


class SimfcsBin(LfdFile):
    """SimFCS raw binary data.

    SimFCS BIN and RAW files contain homogeneous array data of any type and
    shape, stored C-contiguously in little-endian byte order.
    A common format is: shape=(256, 256), dtype='uint16'.

    Examples
    --------
    >>> with SimfcsBin('simfcs.bin', (-1, 256, 256), 'uint16') as f:
    ...     data = f.asarray(memmap=True)
    ...     f.totiff('_simfcs.bin.tif', compression='zlib')
    ...     print(f.shape, data[751, 127, 127])
    (752, 256, 256) 1
    >>> with TiffFile('_simfcs.bin.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(bin|raw)$'

    def _init(self, shape, dtype, offset=0, validate_size=True):
        """Validate file size is a multiple of provided shape and type.

        Parameters
        ----------
        shape : tuple of int
            Shape of array to read from file.
        dtype : numpy.dtype
            Data-type of array in file.
        offset : int, optional
            Position in bytes of array data in file.
            Can be used to skip file header.
        validate_size : bool, optional
            If True, file size must exactly match offset, data shape and dtype.

        """
        if not 0 <= offset <= self._filesize:
            raise LfdFileError(self, 'offset out of range')
        try:
            self.shape = determine_shape(
                shape, dtype, self._filesize - offset, validate=validate_size
            )
        except Exception as exc:
            raise LfdFileError(self) from exc
        self.dtype = numpy.dtype(dtype)
        self._fh.seek(offset)

    def _asarray(self, memmap=False):
        """Return data as array of specified shape and type.

        If memmap, return a read-only memory-map to the data array on disk.

        """
        dtype = '<' + self.dtype.char
        if memmap:
            return numpy.memmap(self._fh, dtype, 'r', self._pos, self.shape)
        data = numpy.fromfile(self._fh, dtype, count=product(self.shape))
        return data.reshape(*self.shape)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        tif.save(self.asarray(), **kwargs)


SimfcsRaw = SimfcsBin


class SimfcsInt(LfdFile):
    """SimFCS intensity image.

    SimFCS INT files contain a single intensity image, stored as 256x256
    float32 or uint16 (older format). The measurement extension is usually
    encoded in the file name.

    Examples
    --------
    >>> with SimfcsInt('simfcs2036.int') as f:
    ...     print(f.asarray()[255, 255])
    3.0
    >>> with SimfcsInt('simfcs1006.int') as f:
    ...     print(f.asarray()[255, 255])
    9

    """

    _filepattern = r'.*\.(int|ac)$'

    def _init(self):
        """Validate file size is 256 KB."""
        if self._filesize == 262144:
            self.dtype = numpy.dtype('<f4')
        elif self._filesize == 131072:
            self.dtype = numpy.dtype('<u2')
        else:
            raise LfdFileError(self, 'file size mismatch')
        self.shape = 256, 256
        self.axes = 'YX'

    def _asarray(self):
        """Return data as 256x256 array of float32 or uint16."""
        return numpy.fromfile(self._fh, self.dtype).reshape(256, 256)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        tif.save(self.asarray(), **kwargs)


class SimfcsIntPhsMod(LfdFile):
    """SimFCS lifetime component images.

    SimFCS INT, PHS and MOD files contain fluorescence lifetime image data
    from frequency-domain measurements.
    Three 256x256 float32 images are stored in separate files:
    intensity (``.int``), phase (``.phs``) and modulation (``.mod``).
    Phase values are in degrees, modulation in percent.
    The measurement extension and channel is often encoded in the file name.

    Examples
    --------
    >>> with SimfcsIntPhsMod('simfcs_1000.phs') as f:
    ...     print(f.asarray().mean(-1).mean(-1))
    [5.72 0.   0.05]

    """

    _filepattern = r'.*\.(int|phs|mod)$'
    _figureargs = {'figsize': (6, 8)}

    def _init(self):
        """Validate file size is 256 KB."""
        if self._filesize != 262144:
            raise LfdFileError(self, 'file size mismatch')
        self.dtype = numpy.dtype('<f4')
        self.shape = 256, 256
        self.axes = 'YX'

    def _components(self):
        """Return possible names of component files."""
        return [(c, self._filename[:-3] + c) for c in ('int', 'phs', 'mod')]

    def _asarray(self):
        """Return image data as (256, 256) shaped array of float32."""
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        tif.save(self.asarray(), **kwargs)

    def _plot(self, figure, **kwargs):
        """Display images stored in files."""
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
    a 8 bytes buffer and the intensity image used for the fit, stored as
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

    Examples
    --------
    >>> with SimfcsFit('simfcs.fit') as f:
    ...     dc_ref, p_fit = f.asarray(size=7)
    ...     f.totiff('_simfcs.fit.tif')
    ...     print(f'{p_fit[6, 1, 1]:.3f} {dc_ref[128, 128]:.2f}')
    0.937 20.23
    >>> with TiffFile('_simfcs.fit.tif') as f:
    ...     assert_array_equal(f.asarray(), dc_ref)

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

    def _init(self):
        """Validate file size is 384 KB."""
        if self._filesize != 393224:
            raise LfdFileError(self, 'file size mismatch')

    def _asarray(self, size=32):
        """Return fit parameters and intensity image as numpy arrays.

        The array shape of the fit parameters will be (size, size, 16).

        Parameters
        ----------
        size : int
            Number of rows and columns of fit parameters array (default: 32).

        """
        if not 0 < size <= 32:
            raise ValueError('size out of range [1..32]')
        p_fit = numpy.fromfile(self._fh, '<f8', 1024 * 16).reshape(1024, 16)
        p_fit = p_fit[: size * size].reshape(size, size, 16)
        self._fh.seek(8, 1)
        dc_ref = numpy.fromfile(self._fh, '<f4', 65536).reshape(256, 256)
        return dc_ref, p_fit

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        dc_ref, p_fit = self.asarray()
        tif.save(dc_ref, description='dc_ref', **kwargs)
        tif.save(p_fit, description='p_fit', **kwargs)

    def _str(self):
        """Return additional information about file."""
        return 'dc_ref: (256, 256) float32\n p_fit: (1024, 16) float64'


class SimfcsCyl(LfdFile):
    """SimFCS orbital tracking data.

    SimFCS CYL files contain intensity data from orbital tracking
    measurements, stored as a uint16 array of shape
    (2 channels, number of orbits, 256 points per orbit).

    The number of channels and points per orbit can be read from the
    associated journal file.

    Examples
    --------
    >>> with SimfcsCyl('simfcs.cyl') as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs.cyl.tif')
    ...     print(f.shape, data[0, -1, :4])
    (2, 3291, 256) [109 104 105 112]
    >>> with TiffFile('_simfcs.cyl.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.cyl$'
    _figureargs = {'figsize': (6, 3)}

    def _init(self, shape=(2, -1, 256)):
        """Verify file size matches shape."""
        channels, orbits, points_per_orbit = shape
        if channels > 2 or channels < 1:
            raise ValueError('channels out of range [1..2]')
        if points_per_orbit > 256 or points_per_orbit < 1:
            raise ValueError('points_per_orbit out of range [1..256]')
        if orbits <= 0:
            orbits = points_per_orbit * channels * 2
            if self._filesize % orbits:
                raise LfdFileError(self, 'invalid shape')
            orbits = int(self._filesize // orbits)
        elif self._filesize != points_per_orbit * orbits * channels * 2:
            raise LfdFileError(self, 'invalid shape')
        self.shape = channels, orbits, points_per_orbit
        self.dtype = numpy.dtype('<u2')

    def _asarray(self):
        """Return data as (channels, -1, points_per_orbit) array of uint16."""
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _plot(self, figure, **kwargs):
        """Display images stored in file."""
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

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        for data in self.asarray():
            tif.save(data, **kwargs)

    def _str(self):
        """Return additional information about file."""
        return 'channels: {}\n orbits: {}\n points_per_orbit: {}'.format(
            *self.shape
        )


class SimfcsRef(LfdFile):
    """SimFCS referenced fluorescence lifetime images.

    SimFCS REF files contain referenced fluorescence lifetime image data.
    Five or more 256x256 float32 images are stored consecutively:
    0) dc - intensity
    1) ph1 - phase of 1st harmonic
    2) md1 - modulation of 1st harmonic
    3) ph2 - phase of 2nd harmonic
    4) md2 - modulation of 2nd harmonic
    Phase values are in degrees, the modulation values are normalized.
    Phase and modulation values may be NaN.

    Examples
    --------
    >>> with SimfcsRef('simfcs.ref') as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs.ref.tif')
    ...     print(f.shape, data[:, 255, 255])
    (5, 256, 256) [301.33  44.71   0.62  68.13   0.32]
    >>> with TiffFile('_simfcs.ref.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.ref$'
    _figureargs = {'figsize': (6, 11)}

    def _init(self):
        """Verify file size is 1280 KB."""
        if self._filesize != 1310720:
            raise LfdFileError(self)
        self.shape = 5, 256, 256
        self.dtype = numpy.dtype('<f4')
        self.axes = 'SYX'

    def _asarray(self):
        """Return images as (-1, 256, 256) shaped array of float32."""
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _plot(self, figure, **kwargs):
        """Display images stored in file."""
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

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=True)
        for image, label in zip(self.asarray(), 'dc ph1 md1 ph2 md2'.split()):
            tif.save(image, description=label, **kwargs)


class SimfcsBh(LfdFile):
    """SimFCS Becker and Hickl fluorescence lifetime histogram.

    SimFCS B&H files contain time-domain fluorescence lifetime histogram data,
    acquired from Becker and Hickl(r) TCSPC cards, or converted from other
    data sources.
    The data are stored consecutively as 256 bins of 256x256 float32 images.
    SimFCS BHZ files are zipped B&H files: a Zip archive containing a single
    B&H file.
    BHZ files are occasionally used to store consecutive 256x256 float32
    images, e.g. volume data.

    Examples
    --------
    >>> with SimfcsBh('simfcs.b&h') as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs.b&h.tif')
    ...     print(f.shape, data[59, 1, 84])
    (256, 256, 256) 12.0
    >>> with TiffFile('_simfcs.b&h.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(b&h)$'

    def _init(self):
        """Verify file size is multiple of 262144."""
        if self._filesize % 262144:
            raise LfdFileError(self)
        self.shape = int(self._filesize // 262144), 256, 256
        self.dtype = numpy.dtype('<f4')
        self.axes = 'QYX'

    def _asarray(self):
        """Return image data as (-1, 256, 256) shaped array of float32."""
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, compression='zlib', contiguous=False)
        data = self.asarray()
        # reshape according to axes provided by user
        axes = parse_kwargs(kwargs, axes=None)['axes']
        if axes:
            shape = {'ZYX': (0, None, 1, 2), 'TYX': (0, None, None, 1, 2)}[
                axes
            ]
            data.shape = [1 if i is None else data.shape[i] for i in shape]
        tif.save(data, **kwargs)


class SimfcsBhz(SimfcsBh):
    """SimFCS compressed Becker and Hickl fluorescence lifetime histogram.

    SimFCS BHZ files contain time-domain fluorescence lifetime histogram data,
    acquired from Becker and Hickl(r) TCSPC cards, or converted from other
    data sources.
    SimFCS BHZ files are zipped B&H files: a Zip archive containing a single
    B&H file.

    Examples
    --------
    >>> with SimfcsBhz('simfcs.bhz') as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs.bhz.tif')
    ...     print(f.shape, data[59, 1, 84])
    (256, 256, 256) 12.0
    >>> with TiffFile('_simfcs.bhz.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(bhz)$'

    def _init(self):
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

    def _asarray(self):
        """Return image data as (256, 256, 256) shaped array of float32."""
        with zipfile.ZipFile(self._fh) as zf:
            data = zf.read(zf.filelist[0])
        return numpy.frombuffer(data, self.dtype).reshape(self.shape).copy()

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, compression='zlib', contiguous=False)
        tif.save(self.asarray(), **kwargs)


class SimfcsB64(LfdFile):
    """SimFCS integer intensity data.

    SimFCS B64 files contain one or more square intensity images, a carpet
    of lines, or a stream of intensity data.
    The intensity data are stored as int16 contiguously after one int32
    defining the image size in x and/or y dimensions if applicable.
    The measurement extension and 'carpet' identifier are usually encoded
    in the file name.

    Examples
    --------
    >>> with SimfcsB64('simfcs.b64') as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs.b64.tif', compression='zlib')
    ...     print(f.shape, data[101, 255, 255])
    (102, 256, 256) 0
    >>> with TiffFile('_simfcs.b64.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.b64$'

    def _init(self, dtype='<i2', maxsize=4096):
        """Read file header."""
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

    def _asarray(self):
        """Return intensity data as 1D, 2D, or 3D array of int16."""
        count = product(self.shape)
        data = numpy.fromfile(self._fh, '<' + self.dtype.char, count=count)
        return data.reshape(*self.shape)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        tif.save(self.asarray(), **kwargs)


def simfcsb64_write(filename, data):
    """Save array of square int16 images to B64 file.

    Refer to the SimfcsB64 class for the B64 file format.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : numpy array
        Data to be saved to file.
        Must be shaped (-1, size, size) and of type int16.

    Examples
    --------
    >>> data = numpy.arange(5*256*256).reshape(5, 256, 256).astype('int16')
    >>> simfcsb64_write('_test.b64', data)
    >>> with SimfcsB64('_test.b64') as f:
    ...     assert_array_equal(f.asarray(), data)

    """
    if data.dtype != 'int16':
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
    zlib deflate compressed stream of one int32 (defining the image size in
    x and y dimensions) and the float32 image data.
    The measurement extension is usually encoded in the file name.
    Update 2020: Sometimes multiple images are stored in I64 files.

    Examples
    --------
    >>> with SimfcsI64('simfcs1000.i64') as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs1000.i64.tif')
    ...     print(f.shape, data[128, 128])
    (256, 256) 12.3125
    >>> with TiffFile('_simfcs1000.i64.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.i64$'

    def _init(self, dtype='<f4', maxsize=1024):
        """Read file header."""
        if not 32 <= self._filesize <= 67108864:  # limit to 64 MB
            raise LfdFileError(self, 'file size out of range')
        size = struct.unpack('<i', self._decompress_header(4))[0]
        if not 2 <= size <= maxsize:
            raise LfdFileError(self, 'image size out of range')
        self.shape = size, size
        self.dtype = numpy.dtype(dtype)
        self.axes = 'YX'

    def _asarray(self):
        """Return data as 2D array of float32."""
        bufsize = product(self.shape) * self.dtype.itemsize + 4
        data = zlib.decompress(self._fh.read(), 15, bufsize)
        data = numpy.frombuffer(data, self.dtype, offset=4)
        data = data.copy()  # make writable
        try:
            data.shape = self.shape[0], self.shape[1]
        except Exception:
            # Z64 format with I64 extension
            data.shape = -1, self.shape[0], self.shape[1]
        return data

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        tif.save(self.asarray(), **kwargs)


def simfcsi64_write(filename, data):
    """Save a single float32 image to I64 file.

    Refer to the SimfcsI64 class for the I64 format.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : numpy array
        Must be shaped (size, size) and of type float32.

    Examples
    --------
    >>> data = numpy.arange(256*256).reshape(256, 256).astype('float32')
    >>> simfcsi64_write('_test.i64', data)
    >>> with SimfcsI64('_test.i64') as f:
    ...     assert_array_equal(f.asarray(), data)

    """
    if data.dtype.char != 'f':
        raise ValueError(f'invalid data type {data.dtype} (must be float32)')
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f'invalid shape {data.shape}')
    data = struct.pack('I', data.shape[0]) + data.tobytes()
    data = zlib.compress(data)
    with open(filename, 'wb') as fh:
        fh.write(data)


class SimfcsZ64(LfdFile):
    """SimFCS compressed image stack.

    SimFCS Z64 files contain stacks of square images such as intensity volumes
    or time-domain fluorescence lifetime histograms acquired from
    Becker and Hickl(r) TCSPC cards.
    The data are stored as zlib deflate compressed stream of two int32
    (defining the image size in x and y dimensions and the number of images)
    and a maximum of 256 square float32 images.
    For file names containing 'allDC', older versions of SimFCS 4 mistakenly
    write the header twice and report the wrong number of images.

    Examples
    --------
    >>> with SimfcsZ64('simfcs.z64') as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs.z64.tif')
    ...     print(f.shape, data[142, 128, 128])
    (256, 256, 256) 2.0
    >>> with TiffFile('_simfcs.z64.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    >>> with SimfcsZ64('simfcs_allDC.z64', doubleheader=True) as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs_allDC.z64.tif')
    ...     print(f.shape, data[128, 128])
    (256, 256) 172.0
    >>> with TiffFile('_simfcs_allDC.z64.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(z64|i64)$'
    _filesizemin = 32

    def _init(self, dtype='<f4', maxsize=(256, 1024), doubleheader=False):
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

    def _asarray(self):
        """Return data as 3D array of float32."""
        bufsize = product(self.shape) * self.dtype.itemsize + 16
        data = zlib.decompress(self._fh.read(), 15, bufsize)
        try:
            data = numpy.frombuffer(
                data, '<' + self.dtype.char, offset=self._skip + 8
            )
            data = data.copy()  # make writable
            return data.reshape(*self.shape)
        except ValueError:
            return data[2:].reshape(*self.shape[1:])

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, compression='zlib', contiguous=False)
        data = self.asarray()
        # reshape according to axes provided by user
        axes = parse_kwargs(kwargs, axes=None)['axes']
        if axes:
            shape = {'ZYX': (0, None, 1, 2), 'TYX': (0, None, None, 1, 2)}[
                axes
            ]
            data.shape = [1 if i is None else data.shape[i] for i in shape]
        tif.save(data, **kwargs)


def simfcsz64_write(filename, data):
    """Save stack of float32 images to Z64 file.

    Refer to the SimfcsZ64 class for the Z64 format.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : numpy array
        Must be shaped (-1, size, size) and of type float32.

    Examples
    --------
    >>> data = numpy.arange(5*256*256).reshape(5, 256, 256).astype('float32')
    >>> simfcsz64_write('_test.z64', data)
    >>> with SimfcsZ64('_test.z64') as f:
    ...     assert_array_equal(f.asarray(), data)

    """
    if data.dtype.char != 'f':
        raise ValueError(f'invalid data type {data.dtype} (must be float32)')
    if data.ndim != 3 or data.shape[1] != data.shape[2]:
        raise ValueError(f'invalid shape {data.shape}')
    data = struct.pack('II', data.shape[2], data.shape[0]) + data.tobytes()
    data = zlib.compress(data)
    with open(filename, 'wb') as fh:
        fh.write(data)


class SimfcsR64(SimfcsRef):
    """SimFCS compressed referenced fluorescence lifetime images.

    SimFCS R64 files contain referenced fluorescence lifetime images.
    The data are stored as a zlib deflate compressed stream of one int32
    (defining the image size in x and y dimensions) and five (or more)
    square float32 images:
    0) dc - intensity
    1) ph1 - phase of 1st harmonic
    2) md1 - modulation of 1st harmonic
    3) ph2 - phase of 2nd harmonic
    4) md2 - modulation of 2nd harmonic
    Phase values are in degrees, the modulation values are normalized.
    Phase and modulation values may be NaN.

    Examples
    --------
    >>> with SimfcsR64('simfcs.r64') as f:
    ...     data = f.asarray()
    ...     f.totiff('_simfcs.r64.tif')
    ...     print(f.shape, data[:, 100, 200])
    (5, 256, 256) [  0.25  23.22   0.64 104.33   2.12]
    >>> with TiffFile('_simfcs.r64.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.r64$'

    def _init(self, dtype='<f4', maxsize=1024):
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

    def _asarray(self):
        """Return data as 3D array of float32."""
        bufsize = product(self.shape) * self.dtype.itemsize + 4
        data = zlib.decompress(self._fh.read(), 15, bufsize)
        data = numpy.frombuffer(data, '<' + self.dtype.char, offset=4)
        data = data.copy()  # make writable
        return data.reshape((-1, self.shape[1], self.shape[2]))

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(
            kwargs, compression='zlib', contiguous=False, metadata=False
        )
        label = 'dc'
        for i, image in enumerate(self.asarray()):
            if i > 0:
                i += 1
                label = f'ph{i // 2}' if i % 2 else f'md{i // 2}'
            tif.save(image, description=label, **kwargs)


def simfcsr64_write(filename, data):
    """Save referenced data to R64 file.

    Refer to the SimfcsR64 class for the format of referenced data.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : numpy array
        Input referenced data.
        Must be shaped (5, size, size) and of type float32.

    Examples
    --------
    >>> data = numpy.arange(5*256*256).reshape(5, 256, 256).astype('float32')
    >>> simfcsr64_write('_test.r64', data)
    >>> with SimfcsR64('_test.r64') as f:
    ...     assert_array_equal(f.asarray(), data)

    """
    if data.dtype.char != 'f':
        raise ValueError(f'invalid data type {data.dtype} (must be float32)')
    if data.ndim != 3 or data.shape[0] < 5 or data.shape[1] != data.shape[2]:
        raise ValueError(f'invalid shape {data.shape}')
    data = struct.pack('I', data.shape[-1]) + data.tobytes()
    data = zlib.compress(data)
    with open(filename, 'wb') as fh:
        fh.write(data)


class SimfcsFbf(LfdFile):
    """FlimBox firmware.

    SimFCS FBF files contain FlimBox device firmwares, stored in binary form
    following a NULL terminated ASCII string containing properties and
    description.

    The properties (lower cased) can be accessed via dictionary interface.

    Examples
    --------
    >>> with SimfcsFbf('simfcs.fbf') as f:
    ...     f['windows']
    ...     f['channels']
    ...     f['secondharmonic']
    ...     'extclk' in f
    16
    2
    0
    True

    """

    _filepattern = r'.*\.(fbf)$'
    _plot = None

    def _init(self, maxheaderlength=1024):
        """Read and parse NULL terminated header string."""
        try:
            # the first 1024 bytes contain the header
            header = self._fh.read(maxheaderlength).split(b'\x00', 1)[0]
        except ValueError as exc:
            raise LfdFileError(self) from exc
        header = bytes2str(header)
        if len(header) != maxheaderlength:
            self._fh.seek(len(header) + 1)
        try:
            header, comment = header.rsplit('/', 1)
            comment = [comment]
        except ValueError:
            comment = []
        self._settings = {}
        for (name, value, unit) in re.findall(
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
            self._settings[name] = (value + unit) if value != '' else True
        self._settings['comment'] = '/'.join(reversed(comment))

    def _asarray(self):
        """Return firmware as binary string."""
        return self._fh.read()

    def _str(self):
        """Return string with header settings."""
        return format_dict(self._settings)

    def __getitem__(self, key):
        return self._settings[key]

    def __contains__(self, key):
        return key in self._settings

    def __len__(self):
        return len(self._settings)

    def __iter__(self):
        return iter(self._settings)


class SimfcsFbd(LfdFile):
    """FlimBox data.

    SimFCS FDB files contain raw data from the FlimBox device, storing a
    stream of 16 bit integers (data words) that can be decoded to photon
    arrival windows, channels, and times.

    This implementation only handles files produced by the first FlimBox
    generation.

    The measurement's frame size, pixel dwell time, number of sampling
    windows, and scanner type are encoded in the last four letters of the
    file name.
    Newer FBD files, where the 3rd character in the file name tag is '0',
    start with the first 1kB of the firmware file used for the measurement,
    followed by 31kB containing a binary record with measurement settings.

    It depends on the application and its setting how to interpret the
    decoded data, e.g. as time series, line scans, or image frames of FCS
    or digital frequency domain fluorescence lifetime measurements.

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

    ``bin``
        Cross correlation phase index (phase histogram bin number).
    ``pcc``
        Cross correlation phase (counter).
    ``pmax``
        Number of entries in cross correlation phase histogram.
    ``pdiv``
        Divisor to reduce number of entries in phase histogram.
    ``win``
        Arrival window.
    ``windows``
        Number of sampling windows.

    Attributes
    ----------
    frame_size : int
        Number of pixels/samples in one line scan, excluding retrace.
    pixel_dwell_time : float
        Number of microseconds the scanner remains at each pixel/sample.
    windows : int
        Number of sampling windows used by FlimBox.
    channels : int
        Number of channels used by FlimBox.
    harmonics : int
        First or second harmonics.
    pmax : int
        Number of entries in cross correlation phase histogram.
    pdiv : int
        Divisor to reduce number of entries in phase histogram.
    laser_frequency : float
        Laser frequency in Hz. Default is 20000000 Hz, the internal
        FlimBox frequency.
    correction_factor : float
        Factor to correct dwell_time/laser_frequency when the scanner
        clock is not known exactly. Default is 1.0.
    scanner : str
        Acquisition software or hardware.
    scanner_line_add : int
        Number of pixels/samples added to each line (for retrace).
    scanner_line_start : int
        Index of first valid pixel/sample in scan line.
    scanner_frame_start : int
        Index of first valid pixel/sample after marker.

    Examples
    --------
    >>> with SimfcsFbd('simfcs$CBCO.fbd') as f:
    ...     bins, times, markers = f.asarray(word_count=500000,
    ...                                      skip_words=1900000)
    >>> print(bins[0, :2], times[:2], markers)
    [53 51] [ 0 42] [ 44097 124815]
    >>> hist = [numpy.bincount(b[b>=0]) for b in bins]
    >>> numpy.argmax(hist[0])
    53

    """

    _filepattern = r'.*\.fbd$'
    _figureargs = {'figsize': (6, 5)}

    _attributes = (
        'frame_size',
        'windows',
        'channels',
        'harmonics',
        'pmax',
        'pdiv',
        'pixel_dwell_time',
        'laser_frequency',
        'correction_factor',
        'scanner',
        'scanner_line_add',
        'scanner_line_start',
        'scanner_frame_start',
    )

    _frame_size = {
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

    _flimbox_settings = {
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

    _histogram_settings = {
        # Map (windows, channels, harmonics) to (pmax, pdiv)
        (2, 2, 1): (256, 4),
        (4, 2, 1): (256, 4),
        (8, 2, 1): (64, 1),
        (8, 4, 1): (64, 1),
        (16, 2, 1): (64, 1),
        # (32, 2, 1): (256, 4), ?
        # (64, 1, 1): (64, 1), ?
        # second harmonics
        (2, 2, 2): (256, 4),
        (4, 2, 2): (128, 2),  # 128, 2?
        (8, 2, 2): (32, 1),
        (8, 4, 2): (32, 1),
        (16, 2, 2): (32, 1),
        # (32, 2, 2): (128, 2), ?
        # (64, 1, 2): (32, 1), ?
    }

    _scanner_settings = {
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
            'A': (6.39, 344, 2, 0),
            'D': (51.21, 176, 8, 414),
            'B': (12.79, 344, 2, 0),
            'E': (102.39, 88, 10, 242),
            'C': (25.61, 344, 2, 0),
            'F': (204.79, 12, 10, 0),
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
        ('pixel_dwell_time_index', '<i4'),
        ('frame_size_index', '<i4'),
        ('line_length', '<i4'),
        ('points_end_of_frame', '<i4'),
        ('x_starting_pixel', '<i4'),
        ('line_integrate', '<i4'),
        ('scanner_index', '<i4'),
        ('synthesizer_index', '<i4'),
        ('windows_index', '<i4'),
        ('channels_index', '<i4'),
        ('_pad1', '<i4'),
        ('line_time', '<f8'),
        ('frame_time', '<f8'),
        ('scanner', '<i4'),
        ('_pad2', '<f4'),
        ('laser_frequency', '<f8'),
        ('laser_factor', '<f8'),
        ('frames_to_average', '<i4'),
        ('enabled_before_start', '<i4'),
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
        4,
        5,
        8,
        10,
        16,
        20,
        32,
        50,
        64,
        100,
        128,
        200,
        256,
        500,
        512,
        1000,
        1,
        2,
    )
    _header_frame_size = (64, 128, 256, 320, 512, 640, 800, 1024)
    _header_bin_pixel_by = (1, 2, 4, 8)
    _header_windows = (4, 8, 16, 32, 64)
    _header_channels = (1, 2, 4)  # '2 to 4 ch', '2 ch 8w Spartan6'
    _header_scanner_types = (
        'None',
        'Native SimFCS, 3-axis card',
        'Olympus FV 1000, NI USB',
        'Zeiss LSM510',
        'Zeiss LSM710',
        'ISS Vista slow scanner',
        'ISS Vista fast scanner',
        'M2 laptop only',
        'IOTech scanner card',
    )
    _header_synthesizer_names = (
        'Internal FLIMbox frequency',
        'Fianium',
        'Spectra Physics MaiTai',
        'Spectra Physics Tsunami',
        'Coherent Chameleon',
    )

    @lazyattr
    def decoder_settings(self):
        """Return dictionary of parameters to decode FlimBox data stream.

        Returns
        -------
        decoder_table : ndarray of int16 and shape (channels, window indices)
            Decoder table, mapping channel and window indices to actual
            arrival windows.
        tcc_mask, tcc_shr : int
            Binary mask and number of bits to right shift in order to extract
            cross correlation time from data word.
        pcc_mask, pcc_shr : int
            Binary mask and number of bits to right shift in order to extract
            cross correlation phase from data word.
        marker_mask, marker_shr: int
            Binary mask and number of bits to right shift in order to extract
            markers from data word.
        win_mask, win_shr : int
            Binary mask and number of bits to right shift in order to extract
            index into lookup table from data word.

        """
        return dict(
            zip(
                (
                    'decoder_table',
                    'tcc_mask',
                    'tcc_shr',
                    'pcc_mask',
                    'pcc_shr',
                    'marker_mask',
                    'marker_shr',
                    'win_mask',
                    'win_shr',
                ),
                getattr(self, f'_w{self.windows}c{self.channels}')(),
            )
        )

    @staticmethod
    def _w4c2():
        # Return parameters to decode 4 windows, 2 channels FlimBox data.
        # fmt: off
        a = [[-1, 0, -1, 1, -1, 2, -1, 3, -1, 0, -1, -1, 1, 0, -1, 2,
              1, 0, 3, 2, 1, 0, 3, 2, 1, -1, 3, 2, -1, -1, 3, -1],
             [-1, -1, 0, -1, 1, -1, 2, -1, 3, 0, -1, -1, 0, 3, -1, 0,
              1, 2, 2, 1, 2, 1, 1, 2, 3, -1, 0, 3, -1, -1, 3, -1]]
        # fmt: on
        a = numpy.array(a, 'int16')
        return a, 0x3FF, 0, 0xFF, 0, 0x400, 10, 0xF800, 11

    @staticmethod
    def _w4c2_():
        # Return parameters to decode 4 windows, 2 channels, second harmonics.
        # fmt: off
        a = [[-1, 0, -1, 1, -1, 2, -1, 3, -1, 0, -1, -1, 1, 0, -1, 2,
              1, 0, 3, 2, 1, 0, 3, 2, 1, -1, 3, 2, -1, -1, 3, -1],
             [-1, -1, 0, -1, 1, -1, 2, -1, 3, 0, -1, -1, 0, 3, -1, 0,
              1, 2, 2, 1, 2, 1, 1, 2, 3, -1, 0, 3, -1, -1, 3, -1]]
        # fmt: on
        a = numpy.array(a, 'int16')
        return a, 0x3FF, 0, 0x3F, 0, 0x400, 10, 0xF800, 11

    @staticmethod
    def _w8c2():
        # Return parameters to decode 8 windows, 2 channels FlimBox data.
        a = numpy.zeros((2, 81), 'int16') - 1
        a[0, 1:9] = range(8)
        a[1, 9:17] = range(8)
        a[:, 17:] = numpy.mgrid[0:8, 0:8].reshape(2, -1)[::-1, :]
        return a, 0xFF, 0, 0x3F, 0, 0x100, 8, 0xFFFF, 9

    @staticmethod
    def _w8c4():
        # Return parameters to decode 8 windows, 4 channels FlimBox data.
        # Not tested.
        a = numpy.zeros((4, 128), 'int16') - 1
        for i in range(128):
            win = i & 0b0000111
            ch0 = (i & 0b0001000) >> 3
            ch1 = (i & 0b0010000) >> 4
            ch2 = (i & 0b0100000) >> 5
            ch3 = (i & 0b1000000) >> 6
            if ch0 + ch1 + ch2 + ch3 != 1:
                continue
            if ch0:
                a[0, i] = win
            elif ch1:
                a[1, i] = win
            elif ch2:
                a[2, i] = win
            elif ch3:
                a[3, i] = win
        return a, 0xFF, 0, 0x3F, 0, 0x100, 8, 0xFFFF, 9

    @staticmethod
    def _w16c1():
        # Return parameters to decode 16 windows, 1 channel FlimBox data.
        # Not tested.
        a = numpy.zeros((1, 32), 'int16') - 1
        for i in range(32):
            win = (i & 0b11110) >> 1
            ch0 = i & 0b00001
            if ch0:
                a[0, i] = win
        return a, 0xFF, 0, 0x3F, 0, 0x100, 8, 0xFFFF, 11

    @staticmethod
    def _w16c2():
        # Return parameters to decode 16 windows, 2 channels FlimBox data.
        # Not tested.
        a = numpy.zeros((2, 64), 'int16') - 1
        for i in range(64):
            win = (i & 0b111100) >> 2
            ch0 = (i & 0b000010) >> 1
            ch1 = i & 0b000001
            if ch0 + ch1 != 1:
                continue
            if ch0:
                a[0, i] = win
            elif ch1:
                a[1, i] = win
        return a, 0xFF, 0, 0x3F, 0, 0x100, 8, 0xFFFF, 10

    @staticmethod
    def _w32c2():
        # Return parameters to decode 32 windows, 2 channels FlimBox data.
        # TODO
        raise NotImplementedError()

    @staticmethod
    def _w64c1():
        # Return parameters to decode 64 windows, 1 channel FlimBox data.
        # TODO
        raise NotImplementedError()

    def _init(self, code=None, **kwargs):
        """Initialize attributes from file name code and additional parameters.

        Parameters
        ----------
        code : str (optional)
            Four character string, encoding frame size (1st char),
            pixel dwell time (2nd char), number of sampling windows (3rd char),
            and scanner type (4th char).
            By default this will be extracted from the file name.
        kwargs : dict
            Optional named parameters to override instance attributes.

        """
        if not code:
            try:
                code = re.search(
                    r'.*\$([A-Z0]{4,4})\.fbd', self._filename, re.IGNORECASE
                ).group(1)
            except AttributeError:
                pass
        if not code:
            code = 'CFCS'  # old FlimBox file ?
            warnings.warn(
                f"failed to detect settings from file name;"
                f"assuming a '{code}' file"
            )

        if code[2] == '0':
            # New FBD file format with header
            # the first 1024 bytes contain the start of a FlimBox firmware file
            with SimfcsFbf(self._fh.name) as fbf:
                self._fbf = fbf._settings
            # the next 31kB contain the binary file header
            self._fh.seek(1024, 0)
            self._header = numpy.fromfile(self._fh, self._header_t, 1)[0]
            hdr = self._header
            if hdr['owner'] != 0:
                raise ValueError(f"unknown header format '{hdr['owner']}'")
            self.frame_size = self._header_frame_size[hdr['frame_size_index']]
            self.scanner = self._header_scanner_types[hdr['scanner_index']]
            self.pixel_dwell_time = self._header_pixel_dwell_time[
                hdr['pixel_dwell_time_index']
            ]
            self.scanner_line_add = int(hdr['line_length']) - self.frame_size
            self.scanner_line_start = int(hdr['x_starting_pixel'])
            self.scanner_frame_start = 0
            self.windows = self._header_windows[hdr['windows_index']]
            self.channels = self._header_channels[hdr['channels_index']]
            self.harmonics = (1, 2)[hdr['second_harmonic']]
            self.laser_frequency = float(hdr['laser_frequency'])
            self.correction_factor = float(hdr['laser_factor'])
            assert self.windows == self._fbf['windows']
            assert self.channels == self._fbf['channels']
            assert hdr['second_harmonic'] == self._fbf['secondharmonic']
            # encoded data start at offset 32768
            self._data_offset = 32768
        else:
            self.frame_size = self._frame_size[code[0]]
            self.scanner = self._scanner_settings[code[3]]['name']
            (
                self.pixel_dwell_time,
                self.scanner_line_add,
                self.scanner_line_start,
                self.scanner_frame_start,
            ) = self._scanner_settings[code[3]][code[1]]
            (
                self.windows,
                self.channels,
                self.harmonics,
            ) = self._flimbox_settings[code[2]]
            self.laser_frequency = 20000000
            self.correction_factor = 1.0
            self._data_offset = 0

        i = self.windows, self.channels, self.harmonics
        try:
            self.pmax, self.pdiv = self._histogram_settings[i]
        except IndexError as exc:
            raise NotImplementedError(
                f'can not handle {self.windows} windows, '
                f'{self.channels} channels, {self.harmonics} harmonics'
            ) from exc

        # override attributes with those specified by user
        for k, v in kwargs.items():
            assert k in self._attributes
            setattr(self, k, v)

        assert self.harmonics in (1, 2)
        self.laser_frequency *= self.harmonics

    def units_per_sample(self):
        """Return number of FlimBox units per scanner sample."""
        return (self.pixel_dwell_time / 1000000) * (
            self.pmax
            / (self.pmax - 1)
            * self.laser_frequency
            * self.correction_factor
        )

    def decode(
        self,
        data=None,
        word_count=-1,
        skip_words=0,
        max_markers=65536,
        **kwargs,
    ):
        """Read FlimBox data stream from file and return decoded data.

        Parameters
        ----------
        data : ndarray or None
            Flimbox data stream. If None (default), read data from file.
        word_count : int (optional)
            Number of data words to process (default: -1, all words).
        skip_words : int (optional)
            Number of data words to skip at beginning of stream (default: 0).
        max_markers : int (optional)
            Maximum number of markers expected in data stream (default: 65536).

        Returns
        -------
        bins : ndarray of int8 and shape (channels, size)
            The cross correlation phase index for all channels and data points.
            A value of -1 means no photon was counted.
        times : ndarray of uint64 or uint32
            The times in FlimBox counter units at each data point.
            A potentially huge array.
        markers : ndarray of ssize_t
            The indices of up markers in the data stream, usually indicating
            frame starts.

        """
        if data is None:
            self._fh.seek(self._data_offset + skip_words * 2, 0)
            data = numpy.fromfile(self._fh, dtype='<u2', count=word_count)
        elif skip_words or word_count != -1:
            if word_count < 0:
                data = data[skip_words:word_count]
            else:
                data = data[skip_words : skip_words + word_count]

        bins_out = numpy.empty((self.channels, data.size), dtype='int8')
        times_out = numpy.empty(data.size, dtype='uint64')
        markers_out = numpy.zeros(max_markers, dtype=numpy.intp)

        simfcsfbd_decode(
            data,
            bins_out,
            times_out,
            markers_out,
            self.windows,
            self.pmax,
            self.pdiv,
            self.harmonics,
            **self.decoder_settings,
        )

        markers_out = markers_out[markers_out > 0]
        if len(markers_out) == max_markers:
            warnings.warn(
                f'number of markers exceeded buffer size: {max_markers}'
            )

        return bins_out, times_out, markers_out

    def frames(
        self,
        decoded,
        select_frames=None,
        aspect_range=(0.8, 1.2),
        frame_cluster=0,
        **kwargs,
    ):
        """Return shape and start/stop indices of scanner frames.

        If unable to detect any frames using the default settings, try to
        determine a correction factor from clusters of frame durations.

        Parameters
        ----------
        decoded : tuple of 3 ndarrays, or None
            Times and markers as returned by the decode function.
            If None, the decode function is called.
        select_frames : slice (optional)
            Specifies which image frames to return (default: all frames).
        aspect_range : tuple (optional)
            Minimum and maximum aspect ratios of valid frames. The default
            lets 1:1 aspects pass.
        frame_cluster : int
            Index of the frame duration cluster to use when calculating
            the correction factor.

        Returns
        -------
        shape : tuple
            Dimensions of scanner frame.
        frame_markers : list of tuples of 2 int
            Start and stop indices of detected image frames.

        """
        if decoded is None:
            decoded = self.decode(**kwargs)
        times, markers = decoded[-2:]

        line_len = self.frame_size + self.scanner_line_add
        line_time = line_len * self.units_per_sample()
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
                if line_num > lines:
                    line_num = lines
            line_num = int(round(line_num))

        if not frame_markers:
            # calculate frame duration clusters, assuming few clusters that
            # are narrower and more separated than cluster_size.
            cluster_size = 1024
            clusters = []
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
            correction_factors = [
                c[0] / (line_time * line_num) for c in clusters
            ]
            # select specified frame cluster
            frame_cluster = min(frame_cluster, len(correction_factors) - 1)
            self.correction_factor = correction_factors[frame_cluster]
            frame_markers = [
                (int(markers[i]), int(markers[i + 1]) - 1)
                for i, c in enumerate(cluster_indices)
                if c == frame_cluster
            ]
            msg = [
                f'no frames detected with default settings.'
                f'Using square shape and correction factor '
                f'{self.correction_factor:.5f}.'
            ]
            if len(correction_factors) > 1:
                msg.append(
                    'The most probable correction factors are: {}'.format(
                        ', '.join(f'{i:.5f}' for i in correction_factors[:4])
                    )
                )
            warnings.warn('\n'.join(msg))

        if not isinstance(select_frames, slice):
            select_frames = slice(select_frames)
        frame_markers = frame_markers[select_frames]
        if not frame_markers:
            raise ValueError('no frames selected')
        return (line_num, line_len), frame_markers

    def asimage(
        self, decoded, frames, integrate_frames=1, square_frame=True, **kwargs
    ):
        """Return image histograms from decoded data and detected frames.

        This function may fail to produce expected results when settings
        were recorded incorrectly, scanner and FlimBox frequencies were out
        of sync, or scanner settings were changed during acquisition.

        Parameters
        ----------
        decoded : tuple of 3 ndarrays, or None
            Bins, times, and markers as returned by the decode function.
            If None, the decode function is called.
        frames : tuple or None
            Scanner_shape and frame_markers as returned by the frames function.
            If None, the frames function is called.
        integrate_frames : int
            Specifies which frames to sum. By default (1), all frames are
            summed into one. If 0, no frames are summed.
        square_frame : bool
            If True (default), return square image (frame_size x frame_size),
            else return full scanner frame.

        Returns
        -------
        result : 5D ndarray of uint16
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
        result = numpy.zeros(shape, dtype='uint16')
        # calculate frame data histogram
        simfcsfbd_histogram(
            bins,
            times,
            frame_markers,
            self.units_per_sample(),
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

    def _asarray(self, **kwargs):
        """Read FlimBox data stream from file and return decoded data."""
        return self.decode(data=None, **kwargs)

    def _plot(self, figure, **kwargs):
        """Plot lifetime histogram for all channels."""
        ax = figure.add_subplot(1, 1, 1)
        ax.set_title(self._filename)
        ax.set_xlim([0, self.pmax // self.pdiv - 1])
        bins, times, markers = self.decode()
        for bins_channel in bins:
            histogram = numpy.bincount(bins_channel[bins_channel >= 0])
            ax.plot(histogram)

    def _str(self):
        """Return additional information about file."""
        return '\n '.join(f'{a}: {getattr(self, a)}' for a in self._attributes)


def simfcsfbd_decode(
    data,
    bins_out,
    times_out,
    markers_out,
    windows,
    pmax,
    pdiv,
    harmonics,
    decoder_table,
    tcc_mask,
    tcc_shr,
    pcc_mask,
    pcc_shr,
    marker_mask,
    marker_shr,
    win_mask,
    win_shr,
):
    """Decode FlimBox data stream.

    See the documentation of the SimfcsFbd class for parameter descriptions
    and the lfdfiles.pyx file for a faster implementation.

    """
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
    markers = numpy.diff(markers.view('int16'))
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
    win = win.astype('int8')
    win[nophoton] = -1
    bins_out[:] = win


def simfcsfbd_histogram(
    bins, times, frame_markers, units_per_sample, scanner_frame_start, out
):
    """Calculate histograms from decoded FlimBox data and frame markers.

    See the documentation of the SimfcsFbd class for parameter descriptions
    and the lfdfiles.pyx file for a much faster implementation.

    """
    # warnings.warn('patience, please.')
    nframes, nchannels, frame_length, nwindows = out.shape
    for f, (j, k) in enumerate(frame_markers):
        f = f % nframes
        t = times[j:k] - times[j]
        t /= units_per_sample  # index into flattened array
        if scanner_frame_start:
            t -= scanner_frame_start
        t = t.astype('uint32')
        numpy.clip(t, 0, frame_length - 1, out=t)
        for c in range(nchannels):
            d = bins[c, j:k]
            for w in range(nwindows):
                x = numpy.where(d == w)[0]
                x = t.take(x)
                x = numpy.bincount(x, minlength=frame_length)
                out[f, c, :, w] += x


try:
    try:
        from ._lfdfiles import simfcsfbd_histogram, simfcsfbd_decode  # noqa
    except ImportError:
        from _lfdfiles import simfcsfbd_histogram, simfcsfbd_decode  # noqa
except ImportError:
    pass

try:
    from ._sflim import sflim_decode
except ImportError:

    def sflim_decode():
        """Raise ImportError: No module names '_sflim'."""
        from ._sflim import sflim_decode  # noqa


class SimfcsGpSeries(LfdFileSequence):
    """SimFCS generalized polarization image series.

    SimFCS GP series contain intensity images from two channels, stored in
    separate SimfcsInt files with consecutive names.

    Examples
    --------
    >>> ims = SimfcsGpSeries('gpint/v*.int')
    >>> ims.axes
    'CI'
    >>> ims = ims.asarray()
    >>> ims.shape
    (2, 135, 256, 256)

    """

    _readfunction = SimfcsInt
    _indexpattern = r'(?P<C>\d)(?P<I>\d+)\.int'


class GlobalsLif(LfdFile):
    """Globals binary lifetime data.

    Globals LIF files contain array and meta data of multiple frequency-domain
    cuvette lifetime measurement, stored as consecutive 472 byte records.
    The number of frequencies per record is limited to 25. The format was
    also used by ISS software.

    Examples
    --------
    >>> with GlobalsLif('globals.lif') as f:
    ...     print(len(f), f[42]['date'], f[42].asarray().shape)
    43 1987.8.8 (5, 11)

    """

    _filepattern = r'.*\.lif$'

    _record_t = numpy.dtype(
        [
            ('_title_len', 'u1'),
            ('title', 'a80'),
            ('number', 'i2'),
            ('frequency', [('_len', 'u1'), ('str', 'a6')], 25),
            ('phase', 'i2', 25),
            ('modulation', 'i2', 25),
            ('deltap', 'i2', 25),
            ('deltam', 'i2', 25),
            ('nanal', 'i2'),
            ('date', 'i2', 3),
            ('time', 'i2', 3),
        ]
    )

    class Record(dict):
        """Record in GlobalsLif files."""

        def asarray(self):
            return numpy.array(
                (
                    self['frequency'],
                    self['phase'],
                    self['modulation'],
                    self['deltap'],
                    self['deltam'],
                )
            )

        def __str__(self):
            return format_dict(self)

    def _init(self):
        """Verify file size and read all records."""
        if self._filesize % 472 or self._filesize // 472 > 1024:
            raise LfdFileError(self)
        self._records = records = []
        for rec in numpy.rec.fromfile(self._fh, self._record_t):
            number = int(rec['number'])
            if number == 0:
                continue
            elif number > 25:
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
                dtype='float64',
            )
            record['phase'] = rec['phase'][:number] / 100.0
            record['modulation'] = rec['modulation'][:number] / 100.0
            record['deltap'] = rec['deltap'][:number] / 100.0
            record['deltam'] = rec['deltam'][:number] / 100.0
            records.append(record)

    def _asarray(self, key=0):
        """Return freq, phi, mod, dp, dm of selected record as numpy array."""
        if self._records:
            return self._records[key].asarray()
        return numpy.empty((5, 0), 'float64')

    def _plot(self, figure, **kwargs):
        """Plot all phase and modulation vs log of frequency."""
        maxplots = 50
        colors = []
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

    def _str(self):
        """Return string with information about file."""
        return f'records: {len(self._records)}'
        # return '\n '.join(f'Record {i}\n{format_dict(r, "  ", trim=0)}'
        #                   for i, r in enumerate(self._records))

    def __getitem__(self, key):
        """Return selected record."""
        return self._records[key]

    def __len__(self):
        """Return number of records."""
        return len(self._records)

    def __iter__(self):
        """Return iterator over records."""
        return iter(self._records)


class GlobalsAscii(LfdFile):
    """Globals ASCII lifetime data.

    Globals ASCII files contain array and meta data of a single frequency
    domain lifetime measurement, stored as human readable ASCII string.
    Consecutive measurements are stored in separate files with increasing file
    extension numbers. The format is also used by ISS and FLOP software.

    The metadata can be accessed via dictionary getitem interface.
    Keys are lower case with spaces replaced by underscores.

    Examples
    --------
    >>> with GlobalsAscii('FLOP.001') as f:
    ...    print(f['experiment'], f.asarray().shape)
    LIFETIME (5, 20)

    """

    _filemode = 'r'
    _filepattern = r'.*\.(\d){3}$'
    _figureargs = {'figsize': (6, 5)}

    def _init(self):
        """Read file and parse into dictionary and data array."""
        if self._fh.read(5) != 'TITLE':
            raise LfdFileError(self)

        self._fh.seek(0)
        content = self._fh.read()
        self._filesize = len(content)
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
        labels = tuple(d.strip().title() for d in match.group(1).split(', '))
        datastr = match.group(2)
        try:
            data = numpy.fromstring(datastr, dtype='float32', sep=' ')
            data = data.reshape((len(labels), -1), order='F')
        except ValueError:
            # try to reconstruct numbers not separated by spaces
            if self._record['experiment'] != 'LIFETIME':
                raise
            data = []
            for line in datastr.splitlines():
                line = line.split('.')
                data.extend(
                    (
                        float(line[0][0:] + '.' + line[1][:7]),
                        float(line[1][7:] + '.' + line[2][:3]),
                        float(line[2][3:] + '.' + line[3][:3]),
                        float(line[3][3:] + '.' + line[4][:3]),
                        float(line[4][3:] + '.' + line[5]),
                    )
                )
            data = numpy.array(data, dtype='float32')
            data = data.reshape((len(labels), -1), order='F')
        self.__data = data
        self._record['data_shape'] = self.__data.shape
        self._record['data'] = labels
        self.close()

    def _asarray(self):
        """Return array data as numpy array."""
        return self.__data.copy()

    def _plot(self, figure, **kwargs):
        """Plot phase and modulation vs log of frequency."""
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

    def _str(self):
        """Return string with information about file."""
        return format_dict(self._record)

    def __getitem__(self, key=0):
        """Return value of key in record."""
        return self._record[key]


class VistaIfli(LfdFile):
    """VistaVision fluorescence lifetime image.

    VistaVision IFLI files contain phasor and lifetime images for several
    time points, channels, slices, and frequencies from analog or digital
    frequency domain fluorescence lifetime measurements.
    After a header of 1024 bytes, the images are stored as two 7 dimensional
    arrays of float32 and shape (time, channel, Z, Y, X, frequency, sample).
    The phasor array has three samples: the average intensity (DC) and the
    real (g) and imaginary (s) parts of the phasor.
    The lifetime array has two samples, the lifetimes calculated from phase
    and modulation (apparent single lifetimes).
    Additional metadata may be stored in an associated XML file.

    Attributes
    ----------
    header : numpy.recarray
        File header.
    shape : tuple of 6 int
        The number of times, channels, Z, Y, X coordinates, and frequencies.

    Examples
    --------
    >>> f = VistaIfli('vista.ifli')
    >>> data = f.asarray()
    >>> print(f.header.frequencies)
    [48000000. 96000000.]
    >>> data.shape
    (1, 2, 1, 128, 128, 2, 3)
    >>> f.totiff('_vista.ifli.tif')
    >>> f.close()
    >>> with TiffFile('_vista.ifli.tif') as f:
    ...     assert_array_equal(f.asarray()[0, 0], data[..., 0, 0])

    """

    _filepattern = r'.*\.ifli$'
    _figureargs = {'figsize': (8, 7.5)}

    def _header_t(self, number_frequencies):
        return numpy.dtype(
            [
                ('signature', 'a12'),  # 'VistaFLImage'
                ('version', 'u1'),
                ('channel_bits', 'u1'),
                ('compression', 'u1'),
                ('dimensions', 'u2', 5),  # TCZYX
                ('boundaries', 'f4', 6),  # x0, x1, y0, y1, z0, z1
                ('coordinate', 'u1'),  # unknown, px, um, mm
                ('pixel_sampling_time', 'f4'),  # milliseconds
                ('pixel_interval_time', 'f4'),
                ('line_interval_time', 'f4'),
                ('frame_interval_time', 'f4'),
                ('number_frequencies', 'i4'),
                ('frequencies', 'f4', number_frequencies),
                ('cross_frequency', 'f4'),
                ('reference_lifetime', 'f4'),  # ns
                ('reference_phasor', 'f4', (number_frequencies, 3)),
            ]
        )

    def _init(self):
        """Read header and metadata from file."""
        if self._fh.read(12) != b'VistaFLImage':
            raise LfdFileError(self)
        self._fh.seek(66)
        numfreq = struct.unpack('<i', self._fh.read(4))[0]
        self._fh.seek(0)
        self.header = h = numpy.rec.fromfile(
            self._fh, self._header_t(numfreq), 1, byteorder='<'
        )[0]
        if h['version'] not in (1, 13):
            log_warning(f'unrecognized VistaIfli file version {h["version"]}')
        if h['compression'] != 0:
            raise NotImplementedError(
                f'VistaIfli compression {h["compression"]} not supported'
            )
        self.shape = tuple(int(i) for i in reversed(h['dimensions'])) + (
            numfreq,
        )
        self.axes = 'TCZYXF'
        self.dtype = numpy.dtype('float32')

    def __getitem__(self, key):
        """Return header attribute."""
        return self.header[key]

    @lazyattr
    def channel_indices(self):
        """Return indices of valid channels."""
        bits = self.header['channel_bits']
        dims = self.header['dimensions'][-2]
        return tuple(i for i in range(dims) if bits & 2 ** i)

    def phasor(self):
        """Return average intensity and phasor coordinates from file.

        The returned array is of type float32 and shape (time, channel, Z, Y,
        X, frequency, sample). The three samples are the average intensity (DC)
        and the real (g) and imaginary (s) parts of the phasor.

        """
        self._fh.seek(1024)
        shape = self.shape + (3,)
        phasor = numpy.fromfile(self._fh, '<f4', product(shape))
        phasor.shape = shape
        return phasor

    def lifetime(self):
        """Return apparent single lifetime data from file.

        The returned array is of type float32 and shape (time, channel, Z, Y,
        X, frequency, sample). It has two samples, the lifetimes calculated
        from phase and modulation.

        """
        self._fh.seek(1024 + product(self.shape) * 3)
        shape = self.shape + (2,)
        lifetime = numpy.fromfile(self._fh, '<f4', product(shape))
        lifetime.shape = shape
        return lifetime

    def _asarray(self):
        """Return phasor data from file."""
        return self.phasor()

    def _plot(self, figure, **kwargs):
        """Display images stored in file."""
        data = self.phasor()
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

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False)
        data = self.phasor()
        data = numpy.moveaxis(data, -2, 0)
        data = numpy.moveaxis(data, -1, 0)
        tif.save(data, **kwargs)


class FlimfastFlif(LfdFile):
    """FlimFast fluorescence lifetime image.

    FlimFast FLIF files contain camera images and metadata of frequency-domain
    fluorescence lifetime measurements.
    A 640 bytes header is followed by a variable number of uint16 images,
    each preceded by a 64 bytes record.

    Attributes
    ----------
    header : numpy.recarray
        File header.
    records : numpy.recarray
        Image headers.

    Examples
    --------
    >>> f = FlimfastFlif('flimfast.flif')
    >>> data = f.asarray()
    >>> f.header.frequency
    80.652
    >>> f.records['phase'][31]
    348.75
    >>> data[31, 219, 299]
    366
    >>> f.totiff('_flimfast.flif.tif')
    >>> f.close()
    >>> with TiffFile('_flimfast.flif.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.flif$'
    _figureargs = {'figsize': (6, 7)}

    _header_t = numpy.dtype(
        [
            ('magic', 'a8'),  # '\211FLF\r\n0\n'
            ('creator', 'a120'),
            ('date', 'a32'),
            ('comments', 'a351'),
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

    def _init(self):
        """Read header and record metadata from file."""
        if not self._fh.read(8) == b'\211FLF\r\n0\n':
            raise LfdFileError(self)
        self._fh.seek(0)
        h = self.header = numpy.rec.fromfile(
            self._fh, self._header_t, 1, byteorder='<'
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
                self._fh, self._record_t, 1, byteorder='<'
            )[0]
            self._fh.seek(stride, 1)
        self.shape = int(h.records), int(h.height), int(h.width)
        self.dtype = numpy.dtype('<u2')
        self.axes = 'PYX'

    def _asarray(self):
        """Return images as (records, height, width) shaped uint16 array."""
        p, h, w = self.shape
        data = numpy.empty((p, h * w), 'uint16')
        self._fh.seek(640 + 64)
        for i in range(p):
            data[i] = numpy.fromfile(self._fh, self.dtype, h * w)
            self._fh.seek(64, 1)
        data.shape = self.shape
        return data

    def _totiff(self, tif, **kwargs):
        """Write phase images and metadata to TIFF file."""
        metadata = {}
        dtypes = {
            'f': float,
            'i': int,
            'u': int,
            'S': lambda x: str(x, 'latin-1'),
        }
        for name, dtype in self.header.dtype.fields.items():
            if name not in ('_', 'magic'):
                dtype = dtypes[dtype[0].kind]
                metadata[name] = dtype(self.header[name])
        for name, dtype in self.records.dtype.fields.items():
            dtype = dtypes[dtype[0].kind]
            metadata[name] = list(dtype(i) for i in self.records[name])
        update_kwargs(kwargs, contiguous=True, metadata=metadata)
        for data in self.asarray():
            tif.save(data, **kwargs)

    def _str(self):
        """Return file header as string."""
        return '\n '.join(
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

    Examples
    --------
    >>> with FlimageBin('flimage.int.bin') as f:
    ...     data = f.asarray()
    ...     f.totiff('_flimage.int.bin.tif')
    ...     print(f.shape, data[:, 219, 299])
    (3, 220, 300) [  1.23 111.8   36.93]
    >>> with TiffFile('_flimage.int.bin.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(int|mod|phi|tmd|tph)\.bin$'
    _figureargs = {'figsize': (6, 7)}

    def _init(self):
        """Verify file size is 264000."""
        if self._filesize != 264000:
            raise LfdFileError(self)
        self.shape = 220, 300
        self.dtype = numpy.dtype('>f4')
        self.axes = 'YX'

    def _components(self):
        """Return possible names of component files."""
        return [
            (c, self._filename[:-7] + c + '.bin')
            for c in ('int', 'phi', 'mod', 'tph', 'tmd')
        ]

    def _asarray(self):
        """Return images as (220, 300) shaped array of float32."""
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _plot(self, figure, **kwargs):
        """Display images stored in files."""
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

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        tif.save(self.asarray(), **kwargs)


class FlieOut(LfdFile):
    """Flie fluorescence lifetime image.

    Flie OUT files contain referenced fluorescence lifetime image data
    from frequency-domain measurements.
    Three 300x220 big-endian float32 images are stored in separate files:
    intensity (``off_*.out``), phase (``phi_*.out``), and modulation
    (``mod_*.out``). Phase values are in degrees, modulation in percent.
    No metadata are available.

    Examples
    --------
    >>> with FlieOut('off_flie.out') as f:
    ...     data = f.asarray()
    ...     f.totiff('_off_flie.out.tif')
    ...     print(f.shape, data[:, 219, 299])
    (3, 220, 300) [91.85 28.24 69.03]
    >>> with TiffFile('_off_flie.out.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'(off|phi|mod)_.*\.out$'
    _figureargs = {'figsize': (6, 7)}

    def _init(self):
        """Verify file size is 264000."""
        if self._filesize != 264000:
            raise LfdFileError(self)
        self.shape = 220, 300
        self.dtype = numpy.dtype('>f4')
        self.axes = 'YX'

    def _components(self):
        """Return possible names of component files."""
        return [(c, c + self._filename[3:]) for c in ('Off', 'Phi', 'Mod')]

    def _asarray(self):
        """Return image data as (220, 300) shaped array of float32."""
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(kwargs, contiguous=False, metadata=None)
        tif.save(self.asarray(), **kwargs)

    def _plot(self, figure, **kwargs):
        """Display images stored in files."""
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
    A number of 256x256 uint16 intensity images is stored consecutively.
    No metadata are available.

    Examples
    --------
    >>> with FliezI16('fliez.i16') as f:
    ...     data = f.asarray()
    ...     f.totiff('_fliez.i16.tif')
    ...     print(f.shape, data[::8, 108, 104])
    (32, 256, 256) [401 538 220 297]
    >>> with TiffFile('_fliez.i16.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.i16$'

    def _init(self):
        """Verify file size is 128 KB."""
        if self._filesize % 131072:
            raise LfdFileError(self)
        self.shape = int(self._filesize // 131072), 256, 256
        self.dtype = numpy.dtype('<u2')
        self.axes = 'IYX'

    def _asarray(self):
        """Return images as (-1, 256, 256) shaped array of uint16."""
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        tif.save(self.asarray(), **kwargs)


class FliezDb2(LfdFile):
    """FLIez double image.

    FLIez DB2 files contain a sequence of images from fluorescence lifetime
    measurements. The modality of the data stored in different files varies:
    phase intensities, average intensities, phase or modulation.
    After a header specifying the 3D data shape, images are stored
    consecutively as float64. No other metadata are available.

    Examples
    --------
    >>> with FliezDb2('fliez.db2') as f:
    ...     data = f.asarray()
    ...     f.totiff('_fliez.db2.tif')
    ...     print(f.shape, data[8, 108, 104])
    (32, 256, 256) 234.0
    >>> with TiffFile('_fliez.db2.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.db2$'

    def _init(self):
        """Read data shape and verify file size."""
        shape = struct.unpack('<iii', self._fh.read(12))
        if self._filesize - 12 != product(shape) * 8:
            raise LfdFileError(self)
        self.shape = shape[::-1]
        self.dtype = numpy.dtype('<f8')
        self.axes = 'IYX'

    def _asarray(self):
        """Return images as 3D array of float64."""
        return numpy.fromfile(self._fh, self.dtype).reshape(self.shape)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        tif.save(self.asarray(), **kwargs)


class BioradPic(LfdFile):
    """Bio-Rad(tm) multi-dimensional data.

    Bio-Rad PIC files contain single-channel volume data or multi-channel
    images.

    Image data in uint8 or uint16 format are stored after a 76 byte header.
    Additional metadata are stored after the image data as 96 byte records
    ("notes").

    No official file format specification is available.
    The header structure was obtained from
    https://forums.ni.com/ni/attachments/ni/200/7567/1/file%20format.pdf
    This implementation does not currently handle multi-file data.

    Examples
    --------
    >>> with BioradPic('biorad.pic') as f:
    ...     data = f.asarray()
    ...     f.totiff('_biorad.pic.tif')
    ...     print(f.shape, data[78, 255, 255])
    (79, 256, 256) 8
    >>> with TiffFile('_biorad.pic.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.pic$'
    _filesizemin = 76
    _figureargs = {'figsize': (8, 6)}

    def _init(self):
        """Read header and validate file Id."""
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
            name=bytes2str(stripnull(name)).strip(),
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
        if npic > 1 and ndims in (2, 3):
            self.axes = 'ZYX' if ndims == 3 else 'CYX'

    def _notes(self):
        """Return metadata from notes records in file."""
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
            note = bytes2str(stripnull(note)).strip()
            # TODO: parse notes to dict
            notes.append((note, notetype, level, x, y))
            if note[:5] == 'AXIS_':
                index, kind, ori, res = note[5:].split(None, 4)[:4]
                if kind == '001':
                    spacing.append(float(res))
                    origin.append(float(ori))
        return notes, tuple(spacing), tuple(origin)

    def _asarray(self):
        """Return image data as array."""
        self._fh.seek(76)
        data = numpy.fromfile(self._fh, self.dtype, product(self.shape))
        return data.reshape(*self.shape)

    def _plot(self, figure, **kwargs):
        """Display images stored in file."""
        data = self._asarray()
        imshow(
            data, figure=figure, title=self._filename, photometric='MINISBLACK'
        )

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        update_kwargs(
            kwargs,
            metadata={
                'axes': self.axes,
                'spacing': self.spacing,
                'origin': self.origin,
            },
        )
        tif.save(self.asarray(), **kwargs)

    def _str(self):
        """Return properties as string."""
        return '\n '.join(
            (
                f'spacing: {self.spacing}',
                f'origin: {self.origin}',
                'header:',
                format_dict(self.header, '  '),
            )
        )


def bioradpic_write(
    filename,
    data,
    axis='Z',
    spacing=None,
    origin=None,
    name=None,
    lens=1,
    mag_factor=1.0,
    image_number=0,
    merged=0,
    edited=0,
    ramp1_min=0,
    ramp1_max=0,
    color1=0,
    ramp2_min=0,
    ramp2_max=0,
    color2=0,
):
    """Save volume or multi-channel data to Bio-Rad(tm) PIC formatted file.

    This implementation does not currently allow writing multi-file datasets,
    advanced metadata, or color palettes.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : array_like
        Input data, two or three-dimensional of uint18 or uint16.
    spacing, origin: tuple of float
        Position and spacing of pixel or voxel in micrometer.

    Refer to the BioradPic header for other parameters.

    Examples
    --------
    >>> data = numpy.arange(1000000).reshape(100, 100, 100).astype('u1')
    >>> bioradpic_write('_test.pic', data)
    >>> with BioradPic('_test.pic') as f:
    ...     assert_array_equal(f.asarray(), data)

    """
    data = numpy.asarray(data)
    if data.ndim not in (2, 3):
        raise ValueError('data must be 2 or 3 dimensional')
    if data.dtype not in ('uint8', 'uint16'):
        raise ValueError('data type must be uint18 or uint16')
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
        1 if data.dtype == 'uint8' else 0,
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
    <http://emdatabank.org/mapformat.html>
    <http://www.ccp4.ac.uk/html/maplib.html>
    <ftp://ftp.wwpdb.org/pub/emdb/doc/map_format/EMDB_mapFormat_v1.0.pdf>

    Attributes
    ----------
    shape : tuple of 3 int
        Shape of data array contained in file.
    start : tuple of 3 int
        Position of first section, row, and column (voxel grid units).
    cell_interval : tuple of 3 int
        Intervals per unit cell repeat along Z, Y, X.
    cell_length : tuple of 3 float
        Unit Cell repeats along Z, Y, X in Angstroms.
    cell_angle : tuple of 3 float
        Unit Cell angles (alpha, beta, gamma) in degrees.
    map_src : tuple of 3 int
        Relationship of Z, Y, X axes to sections, rows, columns
    density_min, density_max, density_mean: float
        Minimum, maximum, average density.
    density_rms : float
        Rms deviation of map from mean density.
    spacegoup : int
        IUCr space group number (1-230).
    skew_matrix, skew_translation : ndarray or None
        Skew matrix and translation vector (if any, else None).
    symboltable : list of byte strings
       Symmetry records as defined in International Tables.

    Examples
    --------
    >>> with Ccp4Map('ccp4.map') as f:
    ...     data = f.asarray()
    ...     f.totiff('_ccp4.map.tif', compression='zlib')
    ...     print(f.shape, data[100, 100, 100])
    (256, 256, 256) 1.0
    >>> with TiffFile('_ccp4.map.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(map|ccp4)$'
    _filesizemin = 1024 + 80
    _dtypes = {0: 'i1', 1: 'i2', 2: 'f4', 4: 'q8', 5: 'i1'}

    def _init(self):
        """Read CCP4 file header and symboltable."""
        header = self._fh.read(1024)
        if header[208:212] not in (b'MAP ', b'PAM\x00', b'MAP\x00'):
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
            self.dtype = byteorder + Ccp4Map._dtypes[mode]
        except KeyError as exc:
            raise LfdFileError(self, f'unknown mode: {mode}') from exc
        self.shape = ns, nr, nc
        self.start = nsstart, nrstart, ncstart
        self.cell_interval = nz, ny, nx
        self.cell_length = z_length, y_length, x_length
        self.cell_angle = alpha, beta, gamma
        self.map_src = maps, mapr, mapc
        if skew_matrix_flag != 0:
            self.skew_translation = numpy.array([T1, T2, T3], 'float64')
            self.skew_matrix = numpy.array(
                [[S11, S12, S13], [S21, S22, S23], [S31, S32, S33]], 'float64'
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

    def _asarray(self, memmap=False):
        """Return volume data as numpy array.

        Parameters
        ----------
        memmap : bool
            If True, use numpy.memmap to read array.

        """
        if memmap:
            return numpy.memmap(
                self._fh,
                dtype=self.dtype,
                mode='r',
                offset=self._pos,
                shape=self.shape,
            )
        data = numpy.fromfile(self._fh, self.dtype, product(self.shape))
        return data.reshape(self.shape)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        tif.save(self.asarray(), **kwargs)

    def _str(self):
        """Return additional information about file."""
        return f'cell length: {self.cell_length}'


def ccp4map_write(
    filename,
    data,
    start=(0, 0, 0),
    cell_interval=None,
    cell_length=None,
    cell_angle=(90, 90, 90),
    map_src=(3, 2, 1),
    density=None,
    density_rms=0,
    spacegroup=1,
    skew_matrix=None,
    skew_translation=None,
    symboltable=b'',
    labels=(b'Created by lfdfiles.py',),
):
    """Save 3D volume data to CCP4 MAP formatted file.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : 3D array_like
        Input volume.

    Refer to the Ccp4Map attributes for other parameters.

    Examples
    --------
    >>> data = numpy.arange(1000000).reshape(100, 100, 100).astype('f4')
    >>> ccp4map_write('_test.ccp4', data)
    >>> with Ccp4Map('_test.ccp4') as f:
    ...     assert_array_equal(f.asarray(), data)

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
    or big-endian byte order, after a header defining data type, endianess,
    and shape.

    Examples
    --------
    >>> with Vaa3dRaw('vaa3d.v3draw') as f:
    ...     data = f.asarray()
    ...     f.totiff('_vaa3d.v3draw.tif')
    ...     print(f.shape, data[2, 100, 100, 100])
    (3, 181, 217, 181) 138
    >>> with TiffFile('_vaa3d.v3draw.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(v3draw|raw)$'
    _filesizemin = 24 + 1 + 2 + 4 * 2  # 2 byte format
    _figureargs = {'figsize': (8, 8)}

    def _init(self):
        """Read header and validate file size."""
        # read 2 byte format header
        header = self._fh.read(24 + 1 + 2 + 4 * 2)
        # first 24 bytes are 'raw_image_stack_by_hpeng'
        if not header.startswith(b'raw_image_stack_by_hpeng'):
            raise LfdFileError(self)
        # next byte is byte order
        byteorder = {b'B': '>', b'L': '<'}[header[24:25]]
        # next two bytes are data itemsize and dtype
        itemsize = struct.unpack(byteorder + 'h', header[25:27])[0]
        self.dtype = byteorder + {1: 'u1', 2: 'u2', 4: 'f4'}[itemsize]
        # next 8 or 16 bytes are data shape
        self.shape = struct.unpack(byteorder + 'hhhh', header[27:])[::-1]
        if self._filesize != len(header) + product(self.shape) * itemsize:
            # 4 byte format
            header += self._fh.read(8)
            self.shape = struct.unpack(byteorder + 'IIII', header[27:])[::-1]
            if self._filesize != len(header) + product(self.shape) * itemsize:
                raise LfdFileError(self, 'file size mismatch')
        self.axes = 'CZYX'

    def _asarray(self):
        """Return data as array."""
        data = numpy.fromfile(self._fh, self.dtype)
        return data.reshape(*self.shape)

    def _plot(self, figure, **kwargs):
        """Display images stored in file."""
        data = self._asarray()
        imshow(
            data, figure=figure, title=self._filename, photometric='MINISBLACK'
        )

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        tif.save(self.asarray(), **kwargs)


def vaa3draw_write(filename, data, byteorder=None, twobytes=False):
    """Save data to Vaa3D RAW binary file(s).

    Refer to the Vaa3dRaw documentation for the v3draw file format.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : array_like
        Input data of type uint8, uint16, or float32.
        Up to 5 dimensions ordered 'TCZYX'.
        Time points are stored in separate files.
    byteorder : str (optional)
        Byte order of data in file.
    twobytes : bool
        If True, store data shape as int16, else uint32 (default).

    Examples
    --------
    >>> data = numpy.arange(1000000).reshape(10, 10, 100, 100).astype('uint16')
    >>> vaa3draw_write('_test.v3draw', data, byteorder='<')
    >>> with Vaa3dRaw('_test.v3draw') as f:
    ...     assert_array_equal(f.asarray(), data)

    """
    data = numpy.array(data, order='C', ndmin=5, copy=False)
    if data.dtype.char not in 'BHf':
        raise ValueError(f'invalid data type {data.dtype}')
    if data.ndim != 5:
        raise ValueError('data must be up to 5 dimensional')
    if byteorder is None:
        byteorder = {'little': '<', 'big': '>'}[sys.byteorder]
    if byteorder not in '><':
        raise ValueError(f'invalid byteorder {byteorder}')
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
        filename = fmt % os.path.splitext(filename)
    for t in range(data.shape[0]):
        with open(filename.format(t=t), 'wb') as fh:
            fh.write(header)
            data[t].astype(dtype).tofile(fh)


class VoxxMap(LfdFile):
    """Voxx color palette.

    Voxx map files contain a single RGB color palette, stored as 256x4
    whitespace separated integers (0..255) in an ASCII file.

    Examples
    --------
    >>> with VoxxMap('voxx.map') as f:
    ...     data = f.asarray()
    ...     f.totiff('_voxx.map.tif')
    ...     print(f.shape, data[100])
    (256, 3) [255 227 155 237]
    >>> with TiffFile('_voxx.map.tif') as f:
    ...     assert_array_equal(f.asarray()[0], data)

    """

    _filemode = 'r'
    _filepattern = r'.*\.map$'
    _figureargs = {'figsize': (6, 1)}

    def _init(self):
        """Verify file starts with numbers."""
        try:
            for i in self._fh.read(32).strip().split():
                if 0 <= int(i) <= 255:
                    continue
                raise ValueError('number out of range')
        except Exception as exc:
            raise LfdFileError(self) from exc
        self.shape = (256, 3)
        self.dtype = numpy.dtype('u1')
        self.axes = 'XS'

    def _asarray(self):
        """Return palette data as (256, 3) shaped array of uint8."""
        self._fh.seek(0)
        data = numpy.fromfile(self._fh, 'u1', 1024, sep=' ')
        return data.reshape(256, 4)

    def _plot(self, figure):
        """Display palette stored in file."""
        pal = self.asarray().reshape(1, 256, -1)
        ax = figure.add_subplot(1, 1, 1)
        ax.set_title(self._filename)
        ax.yaxis.set_visible(False)
        ax.imshow(pal, aspect=20, origin='lower', interpolation='nearest')

    def _totiff(self, tif, **kwargs):
        """Write palette to TIFF file."""
        kwargs.update(photometric='rgb', planarconfig='contig')
        data = numpy.expand_dims(self.asarray(), axis=0)
        tif.save(data, **kwargs)


def voxxmap_write(filename, data):
    """Save data to Voxx map file(s).

    Refer to the VoxxMap documentation for the file format.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : array_like
        Input data of shape (256, 4) and type uint8.

    Examples
    --------
    >>> data = numpy.repeat(numpy.arange(256, dtype='uint8'), 4).reshape(-1, 4)
    >>> voxxmap_write('_test_vox.map', data)
    >>> with VoxxMap('_test_vox.map') as f:
    ...     assert_array_equal(f.asarray(), data)

    """
    data = numpy.array(data, copy=False)
    if data.dtype != 'uint8' or data.shape != (256, 4):
        raise ValueError('not a 256x4 uint8 array')
    numpy.savetxt(filename, data, fmt='%.0f')


class NetpbmFile(LfdFile):
    """Netpbm formatted files.

    Netpbm files contain image data in a variety of formats as specified
    at http://netpbm.sourceforge.net/doc/.

    The following Netpbm and Portable FloatMap formats are supported:
    PBM (bi-level), PGM (grayscale), PPM (color), PAM (arbitrary),
    XV thumbnail (RGB332), PF (float32 RGB), and Pf (float32 grayscale).

    Examples
    --------
    >>> with NetpbmFile('netpbm.pam') as f:
    ...     data = f.asarray()
    ...     f.totiff('_netpbm.pam.tif')
    ...     print(f.shape, data[75, 75, 1])
    (150, 150, 4) 255
    >>> with TiffFile('_netpbm.pam.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(pnm|pbm|pgm|ppm|pam|pfm|xv)$'
    _figureargs = {'figsize': (8, 6)}

    def _init(self, **kwargs):
        """Validate file is a Netpbm file."""
        self._netpbm = None
        if self._fh.read(2) not in (
            b'P1',
            b'P2',
            b'P3',
            b'P4',
            b'P5',
            b'P6',
            b'P7',
            b'PF',
            b'Pf',
        ):
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

    def _asarray(self, **kwargs):
        """Return data from Netpbm file."""
        return self._netpbm.asarray(**kwargs)

    def _plot(self, figure, **kwargs):
        """Display images stored in file."""
        imshow(self.asarray(**kwargs), figure=figure, title=self._filename)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        kwargs.update(metadata=None)
        data = self.asarray()
        if data.ndim > 2 and data.shape[-1] in (3, 4):
            kwargs.update(photometric='rgb', planarconfig='contig')
        tif.save(data, **kwargs)

    def _str(self):
        """Return info about Netpbm file as string."""
        # return str(self._netpbm)

    def __getattr__(self, name):
        """Return attribute from underlying NetpbmFile object."""
        return getattr(self._netpbm, name)


class OifFile(LfdFile):
    """Olympus(r) Image Format files (OIF and OIB).

    Olympus Image Format is the native file format of the Olympus
    FluoView(tm) software for confocal microscopy.

    This class is a light wrapper of the oiffile module. Use the module
    directly to access metadata in OIF and OIB files.

    Examples
    --------
    >>> with OifFile('oiffile.oib') as f:
    ...     print(f.asarray()[2, 100, 100])
    248

    """

    _filepattern = r'.*\.(oib|oif)$'
    _figureargs = {'figsize': (8, 7)}

    def _init(self, **kwargs):
        """Open OIF or OIB file."""
        self._oif = None
        start = self._fh.read(8)
        if (
            start != b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
            and start[:4] != b'\xFF\xFE\x5B\x00'
        ):
            raise LfdFileError(self)

        import oiffile

        try:
            self._oif = oiffile.OifFile(self.filename, **kwargs)
        except Exception as exc:
            raise LfdFileError(self) from exc

        self.axes = self._oif.axes
        self.shape = self._oif.shape
        self.dtype = self._oif.dtype

    def _close(self):
        """Close OifFile instance."""
        if self._oif is not None:
            self._oif.close()
            self._oif = None

    def _asarray(self, **kwargs):
        """Return data from OIF file."""
        return self._oif.asarray(**kwargs)

    def _plot(self, figure, **kwargs):
        """Display images stored in file."""
        imshow(self.asarray(**kwargs), figure=figure, title=self._filename)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        data = self.asarray()
        if data.ndim > 2 and data.shape[-1] in (3, 4):
            kwargs.update(photometric='rgb', planarconfig='contig')
        tif.save(data, **kwargs)

    def _str(self):
        """Return info about OIF as string."""
        # return str(self._oif).split('\n', 2)[-1]

    def __getattr__(self, name):
        """Return attribute from underlying OifFile object."""
        return getattr(self._oif, name)


class CziFile(LfdFile):
    """CZI file.

    Carl Zeiss Image (CZI) is the native file format of the ZEN(r) software
    by Carl Zeiss(r) Microscopy GmbH.

    This class is a light wrapper of the czifile module. Use the module
    directly to access additional image series and metadata in CZI files.

    Examples
    --------
    >>> with CziFile('czifile.czi') as f:
    ...     data = f.asarray()
    ...     f.totiff('_czifile.czi.tif')
    ...     print(f.shape, data[2, 2, 2, 80, 32, 2])
    (3, 3, 3, 250, 200, 3) 255
    >>> with TiffFile('_czifile.czi.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(czi)$'
    _figureargs = {'figsize': (8, 6)}

    def _init(self, **kwargs):
        """Validate file is a CZI file."""
        self._czi = None
        if self._fh.read(10) != b'ZISRAWFILE':
            raise LfdFileError(self)
        self._fh.seek(0)

        import czifile

        try:
            self._czi = czifile.CziFile(self._fh, **kwargs)
        except Exception as exc:
            raise LfdFileError(self) from exc

        self.axes = self._czi.axes
        self.shape = self._czi.shape
        self.dtype = self._czi.dtype

    def _close(self):
        """Close OifFile instance."""
        if self._czi is not None:
            self._czi.close()
            self._czi = None

    def _asarray(self, **kwargs):
        """Return data from CZI file."""
        return self._czi.asarray(**kwargs)

    def _plot(self, figure, **kwargs):
        """Display images stored in file."""
        imshow(self.asarray(**kwargs), figure=figure, title=self._filename)

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        data = self.asarray()
        if data.ndim > 2 and data.shape[-1] <= 4:
            kwargs.update(planarconfig='contig')
        tif.save(data, **kwargs)

    def _str(self):
        """Return CZI info as string."""
        # return str(self._czi).split('\n', 2)[-1]

    def __getattr__(self, name):
        """Return attribute from underlying CziFile object."""
        return getattr(self._czi, name)


class TiffFile(LfdFile):
    """TIFF file.

    TIFF, the Tagged Image File Format, is used to store image and meta data
    from microscopy. Many custom extensions to the standard exist, such as
    LSM, STK, FluoView, MicroManager, ImageJ and OME-TIFF.

    This class is a light wrapper of the tifffile module. Use the module
    directly to access additional image series and metadata in TIFF files.

    Examples
    --------
    >>> with TiffFile('tifffile.tif') as f:
    ...     data = f.asarray()
    ...     f.totiff('_tifffile.tif.tif')
    ...     print(f.shape, data[31, 30, 2])
    (32, 31, 3) 80
    >>> with TiffFile('_tifffile.tif.tif') as f:
    ...     assert_array_equal(f.asarray(), data)

    """

    _filepattern = r'.*\.(tif|tiff|stk|lsm|tf8)$'
    _figureargs = {'figsize': (8, 6)}

    def _init(self, **kwargs):
        """Validate file is a TIFF file."""
        self._tif = None
        if self._fh.read(4) not in (
            b'MM\x00*',
            b'II*\x00',
            b'MM\x00+',
            b'II+\x00',
        ):
            raise LfdFileError(self)
        self._fh.seek(0)

        try:
            self._tif = tifffile.TiffFile(self._fh, **kwargs)
        except Exception as exc:
            raise LfdFileError(self) from exc

        self._series = self._tif.series[0]  # TODO: allow other series
        self.axes = self._series.axes
        self.shape = self._series.shape
        self.dtype = self._series.dtype

    def _close(self):
        """Close OifFile instance."""
        if self._tif is not None:
            self._tif.close()
            self._tif = None

    def _asarray(self, **kwargs):
        """Return data from TIFF file."""
        return self._tif.asarray(**kwargs)

    def _plot(self, figure, **kwargs):
        """Display images stored in file."""
        page = self._series.pages[0]
        imshow(
            self.asarray(**kwargs),
            figure=figure,
            title=self._filename,
            photometric=page.photometric,
            bitspersample=page.bitspersample,
        )

    def _totiff(self, tif, **kwargs):
        """Write image data to TIFF file."""
        data = self.asarray()
        if data.ndim > 2 and data.shape[-1] in (3, 4):
            kwargs.update(photometric='rgb', planarconfig='contig')
        tif.save(data, **kwargs)

    def _str(self):
        """Return TIFF info as string."""
        # return str(self._tif)

    def __getattr__(self, name):
        """Return attribute from underlying TiffFile object."""
        return getattr(self._tif, name)


def convert2tiff(files, verbose=True, skip=None, **kwargs):
    """Convert image data from LfdFile(s) to TIFF files.

    Examples
    --------
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
    files = LfdFileSequence(files, imread=LfdFile).files
    for file in files:
        if verbose:
            print(file, end=' - ')
            sys.stdout.flush()
        for cls in registry:
            try:
                with cls(file, validate=True) as fh:
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
        registry.remove(cls)
        registry.insert(0, cls)


def determine_shape(shape, dtype, size, validate=True, exception=LfdFileError):
    """Validate and return array shape from dtype and data size.

    Parameters
    ----------
    shape : tuple of int
        Shape of array. One shape dimension can be -1. In this case,
        the value is inferred from size and remaining dimensions.
    dtype : numpy.dtype
        Data-type of array.
    size : int
        Size of array data in bytes.
    validate : bool
        If True, 'size' must exactly match 'shape' and 'dtype'.

    Examples
    --------
    >>> determine_shape((-1, 2, 2), 'uint16', 16)
    (2, 2, 2)

    """
    dtype = numpy.dtype(dtype)
    shape = tuple(numpy.array(shape, int))
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


def format_dict(
    adict,
    prefix=' ',
    indent=' ',
    bullets=('', ''),
    excludes=('_',),
    linelen=79,
    trim=1,
):
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


def bytes2str(b, encoding=None, errors='strict'):
    """Return unicode string from encoded bytes."""
    if encoding is not None:
        return b.decode(encoding, errors)
    try:
        return b.decode('utf-8', errors)
    except UnicodeDecodeError:
        return b.decode('cp1252', errors)


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)


def main():
    """Command line usage main function."""
    import click
    from numpy.testing import assert_array_equal  # noqa: used in doctest

    sys.modules[__name__].assert_array_equal = assert_array_equal

    @click.group()
    @click.version_option(version=__version__)
    def cli():
        pass

    @cli.command(help='Run unit tests.')
    def doctest():
        import doctest

        try:
            import lfdfiles.lfdfiles as m
        except ImportError:
            m = None
        try:
            os.chdir('tests')
        except Exception:
            print('Test files not found.')
            return
        numpy.set_printoptions(suppress=True, precision=2)
        doctest.testmod(m, optionflags=doctest.ELLIPSIS)

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
    def convert(format, compress, files):
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
    def view(files):
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


if __name__ == '__main__':
    main()
