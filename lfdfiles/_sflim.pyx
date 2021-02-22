# _sflim.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False

# Copyright (c) 2021, Christoph Gohlke
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

"""Decode Spectral FLIM data from Kintex FLIMbox.

:Authors: Christoph Gohlke and Lorenzo Scipioni

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: BSD 3-Clause

"""

__version__ = '2021.2.22'

import numpy

from cython.parallel import parallel, prange

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

cimport openmp


DEF MASK_PCC = 0b00000000000000000000000011111111
DEF MASK_TCC = 0b00000000000000000000111111111111
DEF MASK_ENA = 0b00000000000000000001000000000000
DEF MASK_WN0 = 0b00000000000000001110000000000000
DEF MASK_PH0 = 0b00000000000000010000000000000000
DEF MASK_PH1 = 0b00000000000000100000000000000000
DEF MASK_WN1 = 0b00000000000111000000000000000000
DEF MASK_WN2 = 0b00000000111000000000000000000000
DEF MASK_PH2 = 0b00000001000000000000000000000000
DEF MASK_PH3 = 0b00000010000000000000000000000000
DEF MASK_WN3 = 0b00011100000000000000000000000000
DEF MASK_ADR = 0b11100000000000000000000000000000

DEF SHR_ENA = 12
DEF SHR_WN0 = 13
DEF SHR_PH0 = 16
DEF SHR_PH1 = 17
DEF SHR_WN1 = 18
DEF SHR_WN2 = 21
DEF SHR_PH2 = 24
DEF SHR_PH3 = 25
DEF SHR_WN3 = 26
DEF SHR_ADR = 29


ctypedef fused uintxx_t:
    uint8_t
    uint16_t


cdef int _decode_address(
    const ssize_t address,
    const uint32_t[::1] data,
    uintxx_t[:, :, :, ::1] sflim,
    const ssize_t size,
    const uint64_t pixeltime,
    const uint64_t enabletime,
    const ssize_t maxframes,
    openmp.omp_lock_t lock
) nogil:
    """Decode single address."""
    cdef:
        ssize_t width = sflim.shape[3]
        ssize_t height = sflim.shape[2]
        ssize_t framesize = width * height
        ssize_t i, c, h, y, x, start, frames, pixelindex
        uint64_t macrotime, macrotime0
        uint32_t d, pcc, tcc, tcc0, enable, ph

    # seek to first frame
    start = 0
    for i in range(size):
        if (data[i] & <uint32_t> MASK_ENA) >> SHR_ENA:
            start = i
            break
    else:
        return -1

    tcc0 = data[start] & <uint32_t> MASK_TCC
    macrotime = tcc0
    macrotime0 = tcc0
    frames = 0

    # loop over all data items
    for i in range(start, size):

        d = data[i]

        if <uint32_t> address != (d & <uint32_t> MASK_ADR) >> SHR_ADR:
            continue

        pcc = d & <uint32_t> MASK_PCC
        tcc = d & <uint32_t> MASK_TCC
        enable = (d & <uint32_t> MASK_ENA) >> SHR_ENA

        if tcc < tcc0:
            macrotime += 4096
        tcc0 = tcc

        if enable and (macrotime - macrotime0 + tcc) > enabletime:
            if frames == maxframes:
                break
            frames += 1
            macrotime0 = macrotime

        pixelindex = <ssize_t> ((macrotime - macrotime0 + tcc) // pixeltime)
        if pixelindex >= framesize:
            # skipped += 1
            continue

        x = pixelindex % width
        y = pixelindex // width

        ph = (d & <uint32_t> MASK_PH0) >> SHR_PH0
        if ph:
            c = address * 4
            h = (pcc + 32 * ((d & <uint32_t> MASK_WN0) >> SHR_WN0)) % 256
            openmp.omp_set_lock(&lock)
            sflim[c, h, y, x] += 1
            openmp.omp_unset_lock(&lock)

        ph = (d & <uint32_t> MASK_PH1) >> SHR_PH1
        if ph:
            c = address * 4 + 1
            h = (pcc + 32 * ((d & <uint32_t> MASK_WN1) >> SHR_WN1)) % 256
            openmp.omp_set_lock(&lock)
            sflim[c, h, y, x] += 1
            openmp.omp_unset_lock(&lock)

        ph = (d & <uint32_t> MASK_PH2) >> SHR_PH2
        if ph:
            c = address * 4 + 2
            h = (pcc + 32 * ((d & <uint32_t> MASK_WN2) >> SHR_WN2)) % 256
            openmp.omp_set_lock(&lock)
            sflim[c, h, y, x] += 1
            openmp.omp_unset_lock(&lock)

        ph = (d & <uint32_t> MASK_PH3) >> SHR_PH3
        if ph:
            c = address * 4 + 3
            h = (pcc + 32 * ((d & <uint32_t> MASK_WN3) >> SHR_WN3)) % 256
            openmp.omp_set_lock(&lock)
            sflim[c, h, y, x] += 1
            openmp.omp_unset_lock(&lock)

    return <int> frames


def sflim_decode(
    const uint32_t[::1] data,
    uintxx_t[:, :, :, ::1] sflim,
    const uint64_t pixeltime,
    uint64_t enabletime=0,
    const ssize_t maxframes=-1,
    const int numthreads=1
):
    """Decode Kintex FLIMBox data to SLIM image array.

    Parameters
    ----------
    data : 1D array of uint32
        Data stream from a Kintex FLIMBox.
    sflim: 4D array of uint8 or uint16
        Initialized array of shape (channels=32, phasebins=256, height, width)
        to which photon counts are added.
    pixeltime : int
        The pixel dwell time in FLIMbox units.
        math.ceil(dwelltime * 256 / 255 * frequency_factor * frequency)
    enabletime : int
        Time in FLIMbox units to wait after detecting an Enable bit before
        detecting the next Enable bit.
    maxframes : int
        Maximum number of image frames to decode.
    numthreads : int
        Number of OpenMP threads to use for decoding addresses in parallel.

    Examples
    --------

    >>> import numpy, math
    >>> data = numpy.fromfile(
    ...     '20210123488_100x_NSC_166_TMRM_4_zoom4000_L115.bin',
    ...      dtype=numpy.uint32
    ... )
    >>> frequency = 78e6
    >>> frequency_factor = 0.9976
    >>> dwelltime = 16e-6
    >>> pixeltime = math.ceil(
    ...     dwelltime * 256 / 255 * frequency_factor * frequency
    ... )
    >>> sflim = numpy.zeros((32, 256, 256, 342), dtype=numpy.uint8)
    >>> sflim_decode(
    ...     data, sflim, pixeltime=pixeltime, maxframes=20, numthreads=6
    ... )
    >>> numpy.unravel_index(numpy.argmax(sflim), sflim.shape)
    (24, 178, 132, 248)

    """
    cdef:
        ssize_t size = data.size
        ssize_t address
        openmp.omp_lock_t lock
        int ret

    if size == 0:
        return
    if sflim.shape[0] != 32 or sflim.shape[1] != 256:
        raise ValueError(
            f'invalid sflim shape {sflim.shape} != (32, 256, height, width)'
        )

    if enabletime == 0:
        enabletime = pixeltime * sflim.shape[3]

    openmp.omp_init_lock(&lock)

    try:
        with nogil, parallel(num_threads=numthreads):
            for address in prange(8):
                ret = _decode_address(
                    address,
                    data,
                    sflim,
                    size,
                    pixeltime,
                    enabletime,
                    maxframes,
                    lock
                )
                if ret < 0:
                    with gil:
                        raise ValueError(
                            f'no start of frame found for address {address}'
                        )
    finally:
        openmp.omp_destroy_lock(&lock)
