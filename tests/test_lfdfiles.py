# tests/test_lfdfiles.py

# Copyright (c) 2012-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the lfdfiles package.

:Version: 2026.4.30

"""

import glob
import itertools
import pathlib

import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from tifffile import TiffFile

import lfdfiles
from lfdfiles import (
    FILE_EXTENSIONS,
    BioradPic,
    Ccp4Map,
    FlieOut,
    FliezDb2,
    FliezI16,
    FlimageBin,
    FlimfastFlif,
    GlobalsAscii,
    GlobalsLif,
    LfdFile,
    LfdFileSequence,
    RawPal,
    SimfcsB64,
    SimfcsBh,
    SimfcsBhz,
    SimfcsBin,
    SimfcsCyl,
    SimfcsFit,
    SimfcsGpSeries,
    SimfcsI64,
    SimfcsInt,
    SimfcsIntPhsMod,
    SimfcsJrn,
    SimfcsR64,
    SimfcsRef,
    SimfcsVpl,
    SimfcsVpp,
    SimfcsZ64,
    Vaa3dRaw,
    VistaIfi,
    VistaIfli,
    VoxxMap,
    bioradpic_write,
    ccp4map_write,
    convert2tiff,
    simfcsb64_write,
    simfcsi64_write,
    simfcsr64_write,
    simfcsz64_write,
    vaa3draw_write,
    voxxmap_write,
)
from lfdfiles.lfdfiles import determine_shape, stripnull

__version__ = lfdfiles.__version__

HERE = pathlib.Path(__file__).parent
DATA = HERE / 'data'
TEMP = HERE / '_temp'
TEMP.mkdir(exist_ok=True)


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert lfdfiles versions match docstrings."""
    ver = ':Version: ' + __version__
    assert __doc__ is not None
    assert lfdfiles.__doc__ is not None
    assert ver in __doc__
    assert ver in lfdfiles.__doc__


def test_lfdfile_open():
    """Test LfdFile.open()."""
    with LfdFile.open(DATA / 'flimfast.flif') as f:
        assert type(f) is FlimfastFlif

    with LfdFile.open(DATA / 'simfcs.ref', validate=False) as f:
        assert type(f) is SimfcsRef

    with LfdFile.open(
        DATA / 'simfcs.bin', validate=False, shape=(-1, 256, 256), dtype='u2'
    ) as f:
        assert type(f) is SimfcsBin

    with pytest.raises(lfdfiles.LfdFileError):
        LfdFile.open(DATA / 'flimfast.flif', registry=[])


def test_lfdfileerror():
    """Test LfdFileError is a ValueError subclass."""
    assert issubclass(lfdfiles.LfdFileError, ValueError)


def test_module_docstring_example():
    """Test module-level docstring example."""
    data = numpy.arange(1000000).reshape((100, 100, 100)).astype('u1')
    bioradpic_write(TEMP / '_biorad.pic', data)

    with BioradPic(TEMP / '_biorad.pic') as f:
        assert f.shape == (100, 100, 100)
        assert f.spacing == (1.0, 1.0, 1.0)
        data = f.asarray()

    with BioradPic(TEMP / '_biorad.pic') as f:
        f.totiff(TEMP / '_biorad.tif', compression='zlib')


def test_lfdfile_flimfast():
    """Test LfdFile with flimfast.flif."""
    with LfdFile(DATA / 'flimfast.flif') as f:
        assert type(f) is FlimfastFlif


def test_lfdfile_simfcs_ref():
    """Test LfdFile with simfcs.ref."""
    with LfdFile(DATA / 'simfcs.ref', validate=False) as f:
        assert type(f) is SimfcsRef


def test_lfdfile_simfcs_bin():
    """Test LfdFile with simfcs.bin."""
    with LfdFile(DATA / 'simfcs.bin', shape=(-1, 256, 256), dtype='u2') as f:
        assert type(f) is SimfcsBin


def test_lfdfile_properties():
    """Test LfdFile properties: filename, size, ndim, repr."""
    with SimfcsI64(DATA / 'simfcs1000.i64') as f:
        assert f.filename == str(DATA / 'simfcs1000.i64')
        assert f.size == 256 * 256
        assert f.ndim == 2
        assert repr(f) == "<SimfcsI64 'simfcs1000.i64'>"


def test_lfdfile_offset():
    """Test LfdFile _offset parameter shifts file read position."""
    prefix = b'\xff' * 16
    data = numpy.arange(256 * 256, dtype='<u2').reshape((256, 256))
    with open(TEMP / '_test_offset.bin', 'wb') as fh:
        fh.write(prefix)
        data.tofile(fh)
    # _offset positions the fh for LfdFile; SimfcsBin also needs its own
    # offset= to skip the prefix when computing shape and seeking for reads
    with SimfcsBin(
        TEMP / '_test_offset.bin',
        _offset=16,
        offset=16,
        shape=(256, 256),
        dtype='u2',
    ) as f:
        assert f._offset == 16
        assert '@16' in repr(f)
        assert_array_equal(f.asarray(), data)


def test_lfdfile_filenotfounderror():
    """Test LfdFile raises FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        LfdFile('_nonexistent_xyz.flif')


def test_lfdfile_validate():
    """Test LfdFile raises LfdFileError on filename pattern mismatch."""
    with pytest.raises(lfdfiles.LfdFileError):
        SimfcsRef('wrong_extension.xyz')


def test_lfdfilesequence():
    """Test LfdFileSequence."""
    ims = LfdFileSequence(
        DATA / 'gpint/v*.int',
        pattern=r'v(?P<Channel>\d)(?P<Image>\d*).int',
        imread=SimfcsInt,
    )
    assert ims.axes == 'CI'
    data = ims.asarray()
    assert data.shape == (2, 135, 256, 256)
    ims.close()


def test_rawpal_rgb():
    """Test RawPal with rgb.pal."""
    with RawPal(DATA / 'rgb.pal') as f:
        assert_array_equal(f.asarray()[100], [16, 255, 239])


def test_rawpal_rgba():
    """Test RawPal with rgba.pal."""
    with RawPal(DATA / 'rgba.pal') as f:
        assert_array_equal(f.asarray()[100], [219, 253, 187, 255])


def test_rawpal_rrggbb():
    """Test RawPal with rrggbb.pal."""
    with RawPal(DATA / 'rrggbb.pal') as f:
        assert_array_equal(f.asarray()[100], [182, 114, 91])


def test_rawpal_rrggbbaa():
    """Test RawPal with rrggbbaa.pal."""
    with RawPal(DATA / 'rrggbbaa.pal') as f:
        assert_array_equal(f.asarray()[100], [182, 114, 91, 170])


def test_rawpal_rrggbbaa_fortran():
    """Test RawPal with rrggbbaa.pal in Fortran order."""
    with RawPal(DATA / 'rrggbbaa.pal') as f:
        assert_array_equal(f.asarray(order='F')[100], [182, 114, 91, 170])


def test_simfcsvpl_simfcs():
    """Test SimfcsVpl with simfcs.vpl."""
    with SimfcsVpl(DATA / 'simfcs.vpl') as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs.vpl.tif')
        assert f.shape == (256, 3)
        assert_array_equal(data[100], [189, 210, 246])
    with TiffFile(TEMP / '_simfcs.vpl.tif') as f:
        assert_array_equal(f.asarray()[0], data)


def test_simfcsvpl_imobj():
    """Test SimfcsVpl with imobj.vpl."""
    with SimfcsVpl(DATA / 'imobj.vpl') as f:
        data = f.asarray()
        f.totiff(TEMP / '_imobj.vpl.tif')
        assert f.shape == (256, 3)
        assert_array_equal(data[100], [0, 254, 27])
    with TiffFile(TEMP / '_imobj.vpl.tif') as f:
        assert_array_equal(f.asarray()[0], data)


def test_simfcsvpp():
    """Test SimfcsVpp."""
    with SimfcsVpp(DATA / 'simfcs.vpp') as f:
        data = f.asarray('nice.vpl')
        f.totiff(TEMP / '_simfcs.vpp.tif')
        assert f.shape == (256, 4)
        assert_array_equal(data[100], [16, 255, 239, 255])
    with TiffFile(TEMP / '_simfcs.vpp.tif') as f:
        assert_array_equal(f.asarray()[35, 0], data)


def test_simfcsjrn():
    """Test SimfcsJrn."""
    with SimfcsJrn(DATA / 'simfcs.jrn', lower=True) as f:
        record: dict[str, object] = f[1]  # type: ignore[assignment]
        assert record['paramters for tracking']['samplimg frequency'] == 15625  # type: ignore[index]


def test_simfcsjrn_raises():
    """Test SimfcsJrn raises ValueError for asarray and totiff."""
    with SimfcsJrn(DATA / 'simfcs.jrn') as f:
        with pytest.raises(ValueError, match='array data'):
            f.asarray()
        with pytest.raises(ValueError, match='image data'):
            f.totiff(TEMP / '_test_jrn.tif')


def test_simfcsbin():
    """Test SimfcsBin."""
    with SimfcsBin(
        DATA / 'simfcs.bin', shape=(-1, 256, 256), dtype='uint16'
    ) as f:
        data = f.asarray(memmap=True)
        f.totiff(TEMP / '_simfcs.bin.tif', compression='zlib')
        assert f.shape == (752, 256, 256)
        assert data[751, 127, 127] == 1
    with TiffFile(TEMP / '_simfcs.bin.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsint_float():
    """Test SimfcsInt with float data."""
    with SimfcsInt(DATA / 'simfcs2036.int') as f:
        assert f.asarray()[255, 255] == 3.0


def test_simfcsint_uint():
    """Test SimfcsInt with uint data."""
    with SimfcsInt(DATA / 'simfcs1006.int') as f:
        assert f.asarray()[255, 255] == 9


def test_simfcsintphsmod():
    """Test SimfcsIntPhsMod."""
    with SimfcsIntPhsMod(DATA / 'simfcs_1000.phs') as f:
        result = f.asarray().mean((1, 2))
        assert_array_almost_equal(result, [5.717, 0, 0.0465], decimal=3)


def test_simfcsfit():
    """Test SimfcsFit."""
    with SimfcsFit(DATA / 'simfcs.fit') as f:
        dc_ref = f.asarray()
        p_fit = f.p_fit(size=7)
        f.totiff(TEMP / '_simfcs.fit.tif')
        assert round(p_fit[6, 1, 1], 3) == 0.937
        assert round(dc_ref[128, 128], 2) == 20.23
    with TiffFile(TEMP / '_simfcs.fit.tif') as f:
        assert_array_equal(f.asarray(), dc_ref)


def test_simfcscyl():
    """Test SimfcsCyl."""
    with SimfcsCyl(DATA / 'simfcs.cyl') as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs.cyl.tif')
        assert f.shape == (2, 3291, 256)
        assert data[0, 1000, 128] == 103
    with TiffFile(TEMP / '_simfcs.cyl.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsref():
    """Test SimfcsRef."""
    with SimfcsRef(DATA / 'simfcs.ref') as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs.ref.tif')
        assert f.shape == (5, 256, 256)
        assert_allclose(
            data[:, 255, 255], [301.3, 44.71, 0.6185, 68.13, 0.3174], rtol=1e-3
        )
    with TiffFile(TEMP / '_simfcs.ref.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsbh():
    """Test SimfcsBh."""
    with SimfcsBh(DATA / 'simfcs.b&h') as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs.b&h.tif')
        assert f.shape == (256, 256, 256)
        assert data[59, 1, 84] == 12.0
    with TiffFile(TEMP / '_simfcs.b&h.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsbhz():
    """Test SimfcsBhz."""
    with SimfcsBhz(DATA / 'simfcs.bhz') as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs.bhz.tif')
        assert f.shape == (256, 256, 256)
        assert data[59, 1, 84] == 12.0
    with TiffFile(TEMP / '_simfcs.bhz.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsb64():
    """Test SimfcsB64."""
    with SimfcsB64(DATA / 'simfcs.b64') as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs.b64.tif', compression='zlib')
        assert f.shape == (102, 256, 256)
        assert data[101, 255, 255] == 0
    with TiffFile(TEMP / '_simfcs.b64.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsb64_write():
    """Test simfcsb64_write."""
    data = numpy.arange(5 * 256 * 256).reshape((5, 256, 256)).astype('int16')
    simfcsb64_write(TEMP / '_test.b64', data)
    with SimfcsB64(TEMP / '_test.b64') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsb64_write_errors():
    """Test simfcsb64_write raises ValueError for invalid input."""
    with pytest.raises(ValueError, match='int16'):
        simfcsb64_write(
            TEMP / '_test.b64', numpy.zeros((5, 256, 256), dtype='f4')
        )
    with pytest.raises(ValueError, match='shape'):
        simfcsb64_write(
            TEMP / '_test.b64', numpy.zeros((256, 128), dtype='int16')
        )


def test_simfcsi64():
    """Test SimfcsI64."""
    with SimfcsI64(DATA / 'simfcs1000.i64') as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs1000.i64.tif')
        assert f.shape == (256, 256)
        assert data[128, 128] == 12.3125
    with TiffFile(TEMP / '_simfcs1000.i64.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsi64_write():
    """Test simfcsi64_write."""
    data = numpy.arange(256 * 256).reshape((256, 256)).astype('f4')
    simfcsi64_write(TEMP / '_test.i64', data)
    with SimfcsI64(TEMP / '_test.i64') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsi64_write_errors():
    """Test simfcsi64_write raises ValueError for invalid input."""
    with pytest.raises(ValueError, match='float32'):
        simfcsi64_write(
            TEMP / '_test.i64', numpy.zeros((256, 256), dtype='u2')
        )
    with pytest.raises(ValueError, match='size, size'):
        simfcsi64_write(
            TEMP / '_test.i64', numpy.zeros((256, 128), dtype='f4')
        )


def test_simfcsz64():
    """Test SimfcsZ64."""
    with SimfcsZ64(DATA / 'simfcs.z64') as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs.z64.tif')
        assert f.shape == (256, 256, 256)
        assert data[142, 128, 128] == 2.0
    with TiffFile(TEMP / '_simfcs.z64.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsz64_doubleheader():
    """Test SimfcsZ64 with doubleheader."""
    with SimfcsZ64(DATA / 'simfcs_allDC.z64', doubleheader=True) as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs_allDC.z64.tif')
        assert f.shape == (256, 256)
        assert data[128, 128] == 172.0
    with TiffFile(TEMP / '_simfcs_allDC.z64.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsz64_write():
    """Test simfcsz64_write."""
    data = numpy.arange(5 * 256 * 256).reshape((5, 256, 256)).astype('f4')
    simfcsz64_write(TEMP / '_test.z64', data)
    with SimfcsZ64(TEMP / '_test.z64') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsz64_write_errors():
    """Test simfcsz64_write raises ValueError for invalid input."""
    with pytest.raises(ValueError, match='float32'):
        simfcsz64_write(
            TEMP / '_test.z64', numpy.zeros((5, 256, 256), dtype='u2')
        )
    with pytest.raises(ValueError, match='shape'):
        simfcsz64_write(
            TEMP / '_test.z64', numpy.zeros((5, 256, 128), dtype='f4')
        )


def test_simfcsr64():
    """Test SimfcsR64."""
    with SimfcsR64(DATA / 'simfcs.r64') as f:
        data = f.asarray()
        f.totiff(TEMP / '_simfcs.r64.tif')
        assert f.shape == (5, 256, 256)
        assert_allclose(
            data[:, 100, 200], [0.25, 23.22, 0.642, 104.3, 2.117], rtol=1e-3
        )
    with TiffFile(TEMP / '_simfcs.r64.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsr64_write():
    """Test simfcsr64_write."""
    data = numpy.arange(5 * 256 * 256).reshape((5, 256, 256)).astype('f4')
    simfcsr64_write(TEMP / '_test.r64', data)
    with SimfcsR64(TEMP / '_test.r64') as f:
        assert_array_equal(f.asarray(), data)


def test_simfcsgpseries():
    """Test SimfcsGpSeries."""
    ims = SimfcsGpSeries(DATA / 'gpint/v*.int')
    assert ims.axes == 'CI'
    data = ims.asarray()
    assert data.shape == (2, 135, 256, 256)


def test_globalslif():
    """Test GlobalsLif."""
    with GlobalsLif(DATA / 'globals.lif') as f:
        assert len(f) == 43
        assert f[42]['date'] == '1987.8.8'
        assert f[42].asarray().shape == (5, 11)


def test_globalslif_raises():
    """Test GlobalsLif.totiff raises ValueError."""
    with GlobalsLif(DATA / 'globals.lif') as f:  # noqa: SIM117
        with pytest.raises(ValueError, match='image data'):
            f.totiff(TEMP / '_test_lif.tif')


def test_globalsascii():
    """Test GlobalsAscii."""
    with GlobalsAscii(DATA / 'FLOP.001') as f:
        assert f['experiment'] == 'LIFETIME'
        assert f.asarray().shape == (5, 20)


def test_vistaifi():
    """Test VistaIfi."""
    f = VistaIfi(DATA / 'vista.ifi')
    data = f.asarray()
    assert f.axes == 'CYX'
    assert f.header.dwelltime == 0.1
    assert data.shape == (2, 128, 128)
    f.totiff(TEMP / '_vista.ifi.tif')
    f.close()
    with TiffFile(TEMP / '_vista.ifi.tif') as tif:
        assert_array_equal(tif.asarray(), data)


def test_vistaifli():
    """Test VistaIfli."""
    f = VistaIfli(DATA / 'vista.ifli')
    data = f.asarray(memmap=True)
    assert f.header['ModFrequency'] == (48000000.0, 96000000.0)
    assert data.shape == (1, 1, 1, 2, 1, 128, 128, 2, 3)
    f.totiff(TEMP / '_vista.ifli.tif')
    f.close()
    with TiffFile(TEMP / '_vista.ifli.tif') as tif:
        assert_array_equal(tif.asarray()[0, 0], data[..., 0, 0])


def test_flimfastflif():
    """Test FlimfastFlif."""
    f = FlimfastFlif(DATA / 'flimfast.flif')
    data = f.asarray()
    assert abs(float(f.header.frequency) - 80.652) < 0.01
    assert float(f.records['phase'][31]) == 348.75
    assert int(data[31, 219, 299]) == 366
    f.totiff(TEMP / '_flimfast.flif.tif')
    f.close()
    with TiffFile(TEMP / '_flimfast.flif.tif') as tif:
        assert_array_equal(tif.asarray(), data)


def test_flimagebin():
    """Test FlimageBin."""
    with FlimageBin(DATA / 'flimage.int.bin') as f:
        data = f.asarray()
        f.totiff(TEMP / '_flimage.int.bin.tif')
        assert f.shape == (3, 220, 300)
        assert_array_almost_equal(
            data[:, 219, 299], [1.23, 111.8, 36.93], decimal=2
        )
    with TiffFile(TEMP / '_flimage.int.bin.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_flieout():
    """Test FlieOut."""
    with FlieOut(DATA / 'off_flie.out') as f:
        data = f.asarray()
        f.totiff(TEMP / '_off_flie.out.tif')
        assert f.shape == (3, 220, 300)
        assert_array_almost_equal(
            data[:, 219, 299], [91.85, 28.24, 69.03], decimal=2
        )
    with TiffFile(TEMP / '_off_flie.out.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_fliezi16():
    """Test FliezI16."""
    with FliezI16(DATA / 'fliez.i16') as f:
        data = f.asarray()
        f.totiff(TEMP / '_fliez.i16.tif')
        assert f.shape == (32, 256, 256)
        assert_array_equal(data[::8, 108, 104], [401, 538, 220, 297])
    with TiffFile(TEMP / '_fliez.i16.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_fliezdb2():
    """Test FliezDb2."""
    with FliezDb2(DATA / 'fliez.db2') as f:
        data = f.asarray()
        f.totiff(TEMP / '_fliez.db2.tif')
        assert f.shape == (32, 256, 256)
        assert data[8, 108, 104] == 234.0
    with TiffFile(TEMP / '_fliez.db2.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_bioradpic():
    """Test BioradPic."""
    with BioradPic(DATA / 'biorad.pic') as f:
        data = f.asarray()
        f.totiff(TEMP / '_biorad.pic.tif')
        assert f.shape == (79, 256, 256)
        assert data[78, 255, 255] == 8
    with TiffFile(TEMP / '_biorad.pic.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_bioradpic_write():
    """Test bioradpic_write."""
    data = numpy.arange(1000000).reshape((100, 100, 100)).astype('u1')
    bioradpic_write(TEMP / '_test.pic', data)
    with BioradPic(TEMP / '_test.pic') as f:
        assert_array_equal(f.asarray(), data)


def test_ccp4map():
    """Test Ccp4Map."""
    with Ccp4Map(DATA / 'ccp4.map') as f:
        data = f.asarray()
        f.totiff(TEMP / '_ccp4.map.tif', compression='zlib')
        assert f.shape == (256, 256, 256)
        assert data[100, 100, 100] == 1.0
    with TiffFile(TEMP / '_ccp4.map.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_ccp4map_write():
    """Test ccp4map_write."""
    data = numpy.arange(1000000).reshape((100, 100, 100)).astype('f4')
    ccp4map_write(TEMP / '_test.ccp4', data)
    with Ccp4Map(TEMP / '_test.ccp4') as f:
        assert_array_equal(f.asarray(), data)


def test_vaa3draw():
    """Test Vaa3dRaw."""
    with Vaa3dRaw(DATA / 'vaa3d.v3draw') as f:
        data = f.asarray()
        f.totiff(TEMP / '_vaa3d.v3draw.tif')
        assert f.shape == (3, 181, 217, 181)
        assert data[2, 100, 100, 100] == 138
    with TiffFile(TEMP / '_vaa3d.v3draw.tif') as f:
        assert_array_equal(f.asarray(), data)


def test_vaa3draw_write():
    """Test vaa3draw_write."""
    data = numpy.arange(1000000).reshape((10, 10, 100, 100)).astype('uint16')
    vaa3draw_write(TEMP / '_test.v3draw', data, byteorder='<')
    with Vaa3dRaw(TEMP / '_test.v3draw') as f:
        assert_array_equal(f.asarray(), data)


def test_vaa3draw_write_multit():
    """Test vaa3draw_write with multiple time points writes separate files."""
    data = numpy.arange(2 * 3 * 4 * 4 * 4, dtype='uint8').reshape(
        (2, 3, 4, 4, 4)
    )
    vaa3draw_write(TEMP / '_test_t.v3draw', data, byteorder='<')
    with Vaa3dRaw(TEMP / '_test_t.t0.v3draw') as f:
        assert_array_equal(f.asarray(), data[0])
    with Vaa3dRaw(TEMP / '_test_t.t1.v3draw') as f:
        assert_array_equal(f.asarray(), data[1])


def test_voxxmap():
    """Test VoxxMap."""
    with VoxxMap(DATA / 'voxx.map') as f:
        data = f.asarray()
        f.totiff(TEMP / '_voxx.map.tif')
        assert f.shape == (256, 4)
        assert_array_equal(data[100], [255, 227, 155, 237])
    with TiffFile(TEMP / '_voxx.map.tif') as f:
        assert_array_equal(f.asarray()[0], data)


def test_voxxmap_write():
    """Test voxxmap_write."""
    data = numpy.repeat(numpy.arange(256, dtype='uint8'), 4).reshape((-1, 4))
    voxxmap_write(TEMP / '_test_vox.map', data)
    with VoxxMap(TEMP / '_test_vox.map') as f:
        assert_array_equal(f.asarray(), data)


def test_convert2tiff():
    """Test convert2tiff."""
    output = pathlib.Path(str(DATA / 'flimfast.flif') + '.tif')
    convert2tiff(DATA / 'flimfast.flif', verbose=False)
    assert output.exists()


def test_simfcsbin_autodetect():
    """Test SimfcsBin auto-detects shape and dtype without hints."""
    with SimfcsBin(DATA / 'simfcs.bin') as f:
        assert f.shape == (752, 256, 256)
        assert f.dtype == numpy.dtype('uint16')


def test_simfcsbin_validate_size_false():
    """Test SimfcsBin opens file with extra bytes using validate_size=False."""
    data = numpy.arange(256 * 256, dtype='<u2').reshape((256, 256))
    with open(TEMP / '_test_extra.bin', 'wb') as fh:
        data.tofile(fh)
        fh.write(b'\x00' * 4)  # 4 extra bytes
    with SimfcsBin(
        TEMP / '_test_extra.bin',
        shape=(256, 256),
        dtype='u2',
        validate_size=False,
    ) as f:
        assert f.shape == (256, 256)
        assert_array_equal(f.asarray(), data)


def test_simfcsb64_carpet():
    """Test SimfcsB64 carpet mode (2D shape when 'carpet' in filename)."""
    import struct

    isize = 100
    orbits = 200
    data = numpy.zeros((orbits, isize), dtype='<i2')
    data[0, 0] = 42
    with open(TEMP / '_test_carpet.b64', 'wb') as fh:
        fh.write(struct.pack('<i', isize))
        data.tofile(fh)
    with SimfcsB64(TEMP / '_test_carpet.b64') as f:
        assert f.shape == (orbits, isize)
        assert f.asarray()[0, 0] == 42


def test_simfcsb64_stream():
    """Test SimfcsB64 stream mode (1D for non-integer number of images)."""
    import struct

    isize = 64
    count = int(isize * isize * 1.5)  # not a whole number of images
    data = numpy.zeros(count, dtype='<i2')
    data[100] = 7
    with open(TEMP / '_test_stream.b64', 'wb') as fh:
        fh.write(struct.pack('<i', isize))
        data.tofile(fh)
    with SimfcsB64(TEMP / '_test_stream.b64') as f:
        assert f.shape == (count,)
        assert f.asarray()[100] == 7


def test_simfcsintphsmod_components():
    """Test SimfcsIntPhsMod components property."""
    with SimfcsIntPhsMod(DATA / 'simfcs_1000.phs') as f:
        assert len(f.components) == 3
        assert [label for label, _ in f.components] == ['int', 'phs', 'mod']
        assert f.shape == (3, 256, 256)


def test_totiff_autoname():
    """Test LfdFile.totiff() with no filename uses auto-naming."""
    import os

    with SimfcsI64(DATA / 'simfcs1000.i64') as f:
        tif_path = f.filename + '.tif'
        if os.path.exists(tif_path):
            os.remove(tif_path)
        f.totiff()
    assert pathlib.Path(tif_path).exists()


def test_simfcsfit_p_fit_errors():
    """Test SimfcsFit.p_fit raises ValueError for out-of-range size."""
    with SimfcsFit(DATA / 'simfcs.fit') as f:
        with pytest.raises(ValueError, match='size out of range'):
            f.p_fit(size=0)
        with pytest.raises(ValueError, match='size out of range'):
            f.p_fit(size=33)


def test_ccp4map_memmap():
    """Test Ccp4Map.asarray with memmap=True returns correct data."""
    with Ccp4Map(DATA / 'ccp4.map') as f:
        data = f.asarray(memmap=True)
        assert f.shape == (256, 256, 256)
        assert data[100, 100, 100] == 1.0
        assert isinstance(data, numpy.memmap)


def test_determine_shape():
    """Test determine_shape."""
    assert determine_shape((-1, 2, 2), 'uint16', 16) == (2, 2, 2)


def test_stripnull():
    """Test stripnull."""
    assert stripnull(b'bytes\x00\x00b') == b'bytes'


@pytest.mark.parametrize(
    'filename',
    itertools.chain.from_iterable(
        glob.glob(f'**/*{ext}', root_dir=DATA, recursive=True)
        for ext in FILE_EXTENSIONS
    ),
)
def test_glob(filename):
    """Test read all LFD files."""
    filename = str(DATA / filename)
    if 'defective' in filename:
        pytest.xfail()
    with LfdFile(filename) as lfd:
        str(lfd)


if __name__ == '__main__':
    import sys
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=lfdfiles')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))

# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
