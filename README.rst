Laboratory for Fluorescence Dynamics (LFD) file formats
=======================================================

Lfdfiles is a Python library and console script for reading, writing,
converting, and viewing many of the proprietary file formats used to store
experimental data at the `Laboratory for Fluorescence Dynamics
<https://www.lfd.uci.edu/>`_.

For command line usage run ``python -m lfdfiles --help``

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: 3-clause BSD

:Version: 2019.4.22

Requirements
------------
* `CPython 2.7 or 3.5+ <https://www.python.org>`_
* `Numpy 1.11.3 <https://www.numpy.org>`_
* `Matplotlib 2.2 <https://pypi.org/project/matplotlib/>`_
  (optional for plotting)
* `Tifffile 2019.1.4 <https://pypi.org/project/tifffile/>`_
  (optional for reading and writing TIFF)
* `Click 7.0 <https://pypi.python.org/pypi/click>`_
  (optional for command line usage)

Revisions
---------
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
    Read TIFF files.
2016.3.29
    Write R64 files.
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

The `Microsoft Visual C++ Redistributable Packages
<https://support.microsoft.com/en-us/help/2977003/
the-latest-supported-visual-c-downloads>`_ are required on Windows.

Many of the LFD's file formats are not documented and might change arbitrarily.
This implementation is mostly based on reverse engineering existing files.
No guarantee can be made as to the correctness of code and documentation.

Experimental data are often stored in plain binary files with metadata
available in separate, human readable journal files (.jrn).

Unless specified otherwise, data are stored in little-endian, C contiguous
order.

Software
--------
The following software is referenced in this module:

(1)  `SimFCS <https://www.lfd.uci.edu/globals/>`_, a.k.a. Globals for
     Images, is software for fluorescence image acquisition, analysis, and
     simulation, developed by Enrico Gratton at UCI.
(2)  `Globals <https://www.lfd.uci.edu/globals/>`_, a.k.a. Globals for
     Spectroscopy, is software for the analysis of multiple files from
     fluorescence spectroscopy, developed by Enrico Gratton at UIUC and UCI.
(3)  ImObj is software for image analysis, developed by LFD at UIUC.
     Implemented on Win16.
(4)  `FlimFast <https://www.lfd.uci.edu/~gohlke/flimfast/>`_ is software for
     frequency-domain, full-field, fluorescence lifetime imaging at video
     rate, developed by Christoph Gohlke at UIUC.
(5)  FLImage is software for frequency-domain, full-field, fluorescence
     lifetime imaging, developed by Christoph Gohlke at UIUC.
     Implemented in LabVIEW.
(6)  FLIez is software for frequency-domain, full-field, fluorescence
     lifetime imaging, developed by Glen Redford at UIUC.
(7)  Flie is software for frequency-domain, full-field, fluorescence
     lifetime imaging, developed by Peter Schneider at MPIBPC.
     Implemented on a Sun UltraSPARC.
(8)  FLOP is software for frequency-domain, cuvette, fluorescence lifetime
     measurements, developed by Christoph Gohlke at MPIBPC.
     Implemented in LabVIEW.
(9)  `VistaVision <http://www.iss.com/microscopy/software/vistavision.html>`_
     is commercial software for instrument control, data acquisition and data
     processing by ISS Inc (Champaign, IL).
(10) `Vaa3D <https://github.com/Vaa3D>`_ is software for multi-dimensional
     data visualization and analysis, developed by the Hanchuan Peng group at
     the Allen Institute.