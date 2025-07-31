Revisions
---------

2025.7.31

- Read variants of SimFCS REF files.
- Drop support for Python 3.10.

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

- Refactor SimfcsFbd initialization.
- Print tracebacks of failing plugins in LfdFile.

2021.7.11

- Calculate pixel_dwell_time and frame_size for FBD files with header.
- Disable simfcsfbd_decode and simfcsfbd_histogram Python code (breaking).

2021.6.25

- Read ISS Vista IFI files.
- Fix reading FBD files with FBF header.
- Fix reading R64 files with excess bytes.
- Fix reading VPL files used by ISS Vista.
- Remove lazyattr.

2021.6.6

- Fix unclosed file warnings.
- Replace TIFF compress with compression parameter (breaking).
- Remove compress option from command line interface (breaking).

2021.2.22

- Add function to decode Spectral FLIM data from Kintex FLIMbox.
- Relax VistaIfli file version check.

2020.9.18

- Drop support for Python 3.6 (NEP 29).
- Support os.PathLike file names.
- Fix writing contiguous series to TIFF files with tifffile >= 2020.9.3.

2020.1.1

- Read CZI files via czifile module.
- Read Olympus Image files via oiffile module.
- Read Netpbm formats via netpbmfile module.
- Add B64, Z64, and I64 write functions.
- Drop support for Python 2.7 and 3.5.

2019.7.2

- Require tifffile 2019.7.2.
- Remove some utility functions.

2019.5.22

- Read and write Bio-Rad(tm) PIC files.
- Read and write Voxx MAP palette files.
- Rename SimfcsMap to Ccp4Map and SimfcsV3draw to Vaa3dRaw (breaking).
- Rename save functions.

2019.4.22

- Fix setup requirements.

2019.1.24

- Add plots for GlobalsLif, SimfcsV3draw, and VistaIfli.
- Support Python 3.7 and numpy 1.15.
- Move modules into lfdfiles package.

2018.5.21

- Update SimfcsB64 to handle carpets and streams.
- Command line interface for plotting and converting to TIFF.
- Registry of LfdFile classes.
- Write image and metadata to TIFF.
- Read TIFF files via tifffile module.

2016.3.29

- Add R64 write function.

2016.3.14

- Read and write Vaa3D RAW volume files.

2015.3.02

- Initial support for plotting.

2015.2.19

- Initial support for new FBD files containing headers.

2014.12.2

- Read B64, R64, I64 and Z64 files (SimFCS version 4).

2014.10.10

- Read SimFCS FIT files.

2014.4.8

- Read and write CCP4 MAP volume files.

2013.8.10

- Read second harmonics FLIMbox data.
- …
