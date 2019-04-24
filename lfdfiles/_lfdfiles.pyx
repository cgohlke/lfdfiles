# -*- coding: utf-8 -*-
# _lfdfiles.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2012-2019, Christoph Gohlke
# Copyright (c) 2012-2019, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
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

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: 3-clause BSD

:Version: 2019.4.22

"""

__version__ = '2019.4.22'


from cython.parallel import parallel, prange

from libc.stdint cimport int8_t, int16_t, uint16_t, uint32_t, uint64_t


ctypedef fused uintxx_t:
    uint32_t
    uint64_t


def simfcsfbd_decode(
    uint16_t[::] data,
    int8_t[:, ::1] bins_out,
    uintxx_t[::1] times_out,
    ssize_t[::1] markers_out,
    int windows,
    int pmax,
    int pdiv,
    int harmonics,
    int16_t[:, ::] decoder_table,
    uint16_t tcc_mask,
    uint32_t tcc_shr,
    uint16_t pcc_mask,
    uint32_t pcc_shr,
    uint16_t marker_mask,
    uint32_t marker_shr,
    uint16_t win_mask,
    uint32_t win_shr
    ):
    """Decode FlimBox data stream.

    See the lfdfiles.SimfcsFbd documentation for parameter descriptions.

    """
    cdef:
        int maxwindex = <int>decoder_table.shape[1]
        ssize_t maxmarker = markers_out.size
        ssize_t nchannel = bins_out.shape[0]
        ssize_t datasize = bins_out.shape[1]
        ssize_t i, j, c
        uint16_t d
        uintxx_t tcc_max, t0, t1
        int m0, m1, pcc, win
        int pmax_win = harmonics * pmax // windows

    if bins_out.shape[0] != decoder_table.shape[0]:
        raise ValueError('shape mismatch between bins and decoder_table')
    if bins_out.shape[1] != times_out.size:
        raise ValueError('shape mismatch between bins and time')

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
                        (pmax-1 - (pcc + win*pmax_win) % pmax) // pdiv)
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
    uintxx_t[::1] times,
    frame_markers,
    double units_per_sample,
    double scanner_frame_start,
    uint16_t[:, :, :, ::1] hist_out
    ):
    """Calculate histograms from decoded FlimBox data and frame markers.

    See the lfdfiles.SimfcsFbd documentation for parameter descriptions.

    """
    cdef:
        ssize_t nframes = hist_out.shape[0]
        ssize_t nchannels = hist_out.shape[1]
        ssize_t framelen = hist_out.shape[2]
        ssize_t nwindows = hist_out.shape[3]
        ssize_t i, j, k, f, c, idx
        uintxx_t t0
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
