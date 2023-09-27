Laboratory for Fluorescence Dynamics (LFD) file formats
=======================================================

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
:License: BSD 3-Clause
:Version: 2023.9.26

Quickstart
----------

Install the lfdfiles package and all dependencies from the
`Python Package Index <https://pypi.org/project/lfdfiles/>`_::

    python -m pip install -U lfdfiles[all]

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

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.5, 3.12.0rc
- `Cython <https://pypi.org/project/cython/>`_ 0.29.36 (build)
- `NumPy <https://pypi.org/project/numpy/>`_ 1.25.2
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2023.9.26 (optional)
- `Czifile <https://pypi.org/project/czifile/>`_ 2019.7.2 (optional)
- `Oiffile <https://pypi.org/project/oiffile/>`_ 2023.8.30 (optional)
- `Netpbmfile <https://pypi.org/project/netpbmfile/>`_ 2023.8.30 (optional)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.7.3
  (optional, for plotting)
- `Click <https://pypi.python.org/pypi/click>`_ 8.1.7
  (optional, for command line apps)

Revisions
---------

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

- Fix type hint issues.
- Add py.typed marker.

2023.8.1

- Specify encoding of text files.
- Fix linting issues.

2023.4.20

- Improve type hints.
- Drop support for Python 3.8 and numpy < 1.21 (NEP29).

2022.9.29

- Fix setup.py.

2022.9.20

- Update metadata.

2022.6.10

- Fix LfdFileSequence with tifffile 2022.4.22.
- Add fbd2b64 conversion function and script.
- Add decoder for 32-bit, 8 windows, 4 channels FLIMbox data from Spartan-6.
- Convert docstrings to Google style with Sphinx directives.

2022.2.2

- Add type hints.
- SimfcsFit.asarray returns dc_ref only; use p_fit for fit params (breaking).
- Remove additional positional arguments to LfdFile init (breaking).
- Guess SimfcsBin shape and dtype if not provided (breaking).
- Use TiffWriter.write instead of deprecated save.
- Drop support for Python 3.7 and NumPy < 1.19 (NEP29).

2021.7.15

- …

Refer to the CHANGES file for older revisions.

Notes
-----

The API is not stable yet and might change between revisions.

Python <= 3.8 is no longer supported. 32-bit versions are deprecated.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2015-2022
<https://support.microsoft.com/en-us/help/2977003/
the-latest-supported-visual-c-downloads>`_ is required on Windows.

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
(100, 100, 100)
(1.0, 1.0, 1.0)

Convert the PIC file to a compressed TIFF file:

>>> with BioradPic('_biorad.pic') as f:
...     f.totiff('_biorad.tif', compression='zlib')
