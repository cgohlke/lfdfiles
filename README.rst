Laboratory for Fluorescence Dynamics (LFD) file formats
=======================================================

Lfdfiles is a Python library and console script for reading, writing,
converting to TIFF, and viewing many of the proprietary file formats used
to store experimental data and metadata at the
`Laboratory for Fluorescence Dynamics <https://www.lfd.uci.edu/>`_.
For example:

* SimFCS VPL, VPP, JRN, BIN, INT, CYL REF, BH, BHZ FBF, FBD, B64, I64, Z64, R64
* GLOBALS LIF, ASCII
* CCP4 MAP
* Vaa3D RAW
* Bio-Rad(r) PIC
* Vista IFLI, IFI
* FlimFast FLIF

For command line usage run ``python -m lfdfiles --help``

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2022.2.2

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.8.10, 3.9.9, 3.10.1, 64-bit <https://www.python.org>`_
* `Cython 0.29.27 <https://cython.org>`_ (build)
* `Numpy 1.21.5 <https://pypi.org/project/numpy/>`_
* `Tifffile 2021.11.2  <https://pypi.org/project/tifffile/>`_  (optional)
* `Czifile 2019.7.2 <https://pypi.org/project/czifile/>`_ (optional)
* `Oiffile 2021.6.6 <https://pypi.org/project/oiffile />`_ (optional)
* `Netpbmfile 2021.6.6 <https://pypi.org/project/netpbmfile />`_ (optional)
* `Matplotlib 3.4.3 <https://pypi.org/project/matplotlib/>`_
  (optional for plotting)
* `Click 8.0 <https://pypi.python.org/pypi/click>`_
  (optional for command line usage)

Revisions
---------
2022.2.2
    Add type hints.
    SimfcsFit.asarray returns dc_ref only; use p_fit for fit params (breaking).
    Remove additional positional arguments to LfdFile init (breaking).
    Guess SimfcsBin shape and dtype if not provided (breaking).
    Use TiffWriter.write instead of deprecated save.
    Drop support for Python 3.7 and numpy < 1.19 (NEP29).
2021.7.15
    Refactor SimfcsFbd initialization.
    Print tracebacks of failing plugins in LfdFile.
2021.7.11
    Calculate pixel_dwell_time and frame_size for FBD files with header.
    Disable simfcsfbd_decode and simfcsfbd_histogram Python code (breaking).
2021.6.25
    Read ISS Vista IFI files.
    Fix reading FBD files with FBF header.
    Fix reading R64 files with excess bytes.
    Fix reading VPL files used by ISS Vista.
    Remove lazyattr.
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
2019.7.2
   Require tifffile 2019.7.2.
   Remove some utility functions.
2019.5.22
    Read and write Bio-Rad(tm) PIC files.
    Read and write Voxx MAP palette files.
    Rename SimfcsMap to Ccp4Map and SimfcsV3draw to Vaa3dRaw (breaking).
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
    Read second harmonics FLIMbox data.

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
11. `Voxx <https://voxx.sitehost.iu.edu/>`_ is a volume rendering program
    for 3D microscopy, developed by Jeff Clendenon et al. at the Indiana
    University.
12. `CCP4 <https://www.ccp4.ac.uk/>`_, the Collaborative Computational Project
    No. 4, is software for macromolecular X-Ray crystallography.
