# _lfdfiles.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False

# Copyright (c) 2012-2023, Christoph Gohlke
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

"""Fast implementations for the lfdfiles module.

:Author: Christoph Gohlke
:License: BSD 3-Clause

"""

__version__ = '2023.4.20'


from cython.parallel import parallel, prange

from libc.stdint cimport int8_t, int16_t, uint8_t, uint16_t, uint32_t, uint64_t

cimport openmp

ctypedef fused data_t:
    uint16_t
    uint32_t

ctypedef fused times_t:
    uint32_t
    uint64_t

ctypedef fused sflim_t:
    uint8_t
    uint16_t


def simfcsfbd_decode(
    data_t[::] data,
    int8_t[:, ::1] bins_out,
    times_t[::1] times_out,
    ssize_t[::1] markers_out,
    int windows,
    int pdiv,
    int harmonics,
    int16_t[:, ::] decoder_table,
    data_t tcc_mask,
    uint32_t tcc_shr,
    data_t pcc_mask,
    uint32_t pcc_shr,
    data_t marker_mask,
    uint32_t marker_shr,
    data_t win_mask,
    uint32_t win_shr
    ):
    """Decode FLIMbox data stream.

    Parameters:
        data (numpy.ndarray):
            FLIMbox data stream without header.
            An `uint16` (16-bit FLIMbox) or `uint32` (32-bit FLIMbox) array.
        bins_out (numpy.ndarray):
            `ssize_t` array of shape `(channels, data.size)`, where decoded
            cross correlation phase indices are returned.
            A value of -1 means no photon was counted.
        times_out (numpy.ndarray):
            `uint32` or `uint64` array of length `data.size`, where
            times in FLIMbox counter units at each data point are returned.
        markers_out (numpy.ndarray):
            A `ssize_t` array of appropriate length, where indices into
            `times_out` are returned for all data words with markers enabled.
        windows (int):
            Number of sampling windows.
        pdiv (int):
            Divisor to reduce number of entries in phase histogram.
        harmonics (int):
            Decode first or second harmonics.
        decoder_table (numpy.ndarray):
            Mapping of channel and window indices to actual arrival windows.
            An `int16` array of shape
            `(channels, max of cross correlation phase)`.
        tcc_mask (int):
            Binary mask to extract cross correlation time from data word.
        tcc_shr (int):
            Number of bits to right shift masked cross correlation time.
        pcc_mask (int):
            Binary mask to extract cross correlation phase from data word.
        pcc_shr (int):
            Number of bits to right shift masked cross correlation phase.
        marker_mask (int):
            Binary mask to extract markers from data word.
        marker_shr (int):
            Number of bits to right shift masked marker.
        win_mask (int):
            Binary mask to extract index into `decoder_table` from data word.
        win_shr (int):
            Number of bits to right shift masked index into `decoder_table`.

    """
    cdef:
        int maxwindex = <int>decoder_table.shape[1]
        ssize_t maxmarker = markers_out.size
        ssize_t nchannel = bins_out.shape[0]
        ssize_t datasize = bins_out.shape[1]
        ssize_t i, j, c
        data_t d
        times_t tcc_max, t0, t1
        int m0, m1, pcc, win
        int pmax = ((pcc_mask >> pcc_shr) + 1) // harmonics
        int pmax_win = harmonics * pmax // windows

    if bins_out.shape[0] != decoder_table.shape[0]:
        raise ValueError('shape mismatch between bins and decoder_table')
    if bins_out.shape[1] != times_out.size:
        raise ValueError('shape mismatch between bins and time')
    if pmax <= 1 or pmax_win < 1 or pdiv < 1:
        raise ValueError('invalid parameters')

    # calculate cross correlation phase index
    for c in prange(nchannel, nogil=True):
        for i in range(datasize):
            d = data[i]
            pcc = <int>((d & pcc_mask) >> pcc_shr)
            win = <int>((d & win_mask) >> win_shr)
            if win < maxwindex:
                win = <int>(decoder_table[c, win])
                if win >= 0:
                    bins_out[c, i] = <int8_t>(
                        (pmax-1 - (pcc + win * pmax_win) % pmax) // pdiv)
                else:
                    bins_out[c, i] = -1  # no event
            else:
               bins_out[c, i] = -2  # should never happen

    # record up-markers and absolute time
    tcc_max = (tcc_mask >> tcc_shr) + 1
    j = 0
    m0 = <int>(data[0] & marker_mask)
    t0 = (data[0] & tcc_mask) >> tcc_shr
    times_out[0] = 0
    for i in range(1, datasize):
        d = data[i]
        # detect up-markers
        if j < maxmarker:
            m1 = <int>(d & marker_mask)
            if m1 > m0:
                markers_out[j] = i
                j += 1
            m0 = m1
        # cumulative sum of differences of cross correlation time
        t1 = t0
        t0 = (d & tcc_mask) >> tcc_shr
        if t0 > t1:
            times_out[i] = times_out[i-1] + (t0 - t1)
        elif t0 == 0:
            times_out[i] = times_out[i-1] + (tcc_max - t1)
        else:
            # is this supposed to happen?  0 < t0 <= t1
            times_out[i] = times_out[i-1] + (tcc_max - t1) + t0


def simfcsfbd_histogram(
    int8_t[:, ::1] bins,
    times_t[::1] times,
    frame_markers,
    double units_per_sample,
    double scanner_frame_start,
    uint16_t[:, :, :, ::1] hist_out
    ):
    """Calculate histograms from decoded FLIMbox data and frame markers.

    Parameters:
        bins (numpy.ndarray):
            Cross correlation phase index for all channels and data points.
            A `int8` array of shape `(channels, data.size)`.
            A value of -1 means no photon was counted.
        times (numpy.ndarray):
            Times in FLIMbox counter units at each data point.
            An `uint64` or `uint32` array of length `data.size`.
        frame_markers (list[tuple[int, int]]):
            Start and stop indices of detected image frames.
        units_per_sample (float):
            Number of FLIMbox units per scanner sample.
        scanner_frame_start (int):
            Index of first valid pixel/sample after marker.
        hist_out (numpy.ndarray):
            Initialized `uint16` array of shape `(number of frames, channels,
            detected line numbers, frame_size, histogram bins)`,
            where computed histogram will be stored.

    """
    cdef:
        ssize_t nframes = hist_out.shape[0]
        ssize_t nchannels = hist_out.shape[1]
        ssize_t framelen = hist_out.shape[2]
        ssize_t nwindows = hist_out.shape[3]
        ssize_t i, j, k, f, c, idx
        times_t t0
        int8_t w

    if bins.shape[0] != hist_out.shape[1]:
        raise ValueError('shape mismatch between bins and hist_out')
    if bins.shape[1] != times.shape[0]:
        raise ValueError('shape mismatch between bins and times')

    units_per_sample = 1.0 / units_per_sample
    for f, (j, k) in enumerate(frame_markers):
        f = f % nframes
        t0 = times[j]
        for c in prange(nchannels, nogil=True):
            for i in range(j, k):
                idx = <ssize_t>(<double>(times[i] - t0) * units_per_sample
                                - scanner_frame_start)
                if idx >= 0 and idx < framelen:
                    w = bins[c, i]
                    if w >= 0:
                        hist_out[f, c, idx, w] += 1


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


def sflim_decode(
    const uint32_t[::1] data,
    sflim_t[:, :, :, ::1] sflim,
    const uint64_t pixeltime,
    uint64_t enabletime=0,
    const ssize_t maxframes=-1,
    const int numthreads=1
):
    """Decode Kintex FLIMbox data to SLIM image array.

    Parameters:
        data (numpy.ndarray):
            Data stream from Kintex FLIMbox. A `uint32` array.
        sflim (numpy.ndarray):
            Initialized `uint8` or `uint16` array of shape
            `(channels=32, phasebins=256, height, width)`
            to which photon counts are added.
        pixeltime (int):
            Pixel dwell time in FLIMbox units.
            ``math.ceil(dwelltime * 256 / 255 * frequency_factor * frequency)``
        enabletime (int):
            Time in FLIMbox units to wait after detecting enable bit before
            detecting next enable bit.
        maxframes (int):
            Maximum number of image frames to decode.
        numthreads (int):
            Number of OpenMP threads to use for decoding addresses in parallel.

    Examples:
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


cdef int _decode_address(
    const ssize_t address,
    const uint32_t[::1] data,
    sflim_t[:, :, :, ::1] sflim,
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