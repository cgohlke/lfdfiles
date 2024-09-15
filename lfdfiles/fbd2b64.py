#!/usr/bin/env python3
# lfdfiles/fbd2b64.py
# mypy: allow-untyped-defs, allow-untyped-calls

"""Command line script to convert FLIMbox data file to SimFCS B64 files.

::

  Usage: fbd2b64 [OPTIONS] [FBDFILE]

  Options:
    --version                       Show the version and exit.
    --show                          Plot decoded FLIMbox data.
    --quiet                         Do not print information about file
                                    and conversion.
    --integrate_frames              Sum all frames.
    --full_frame                    Return full scanner frames.
    --pdiv INTEGER RANGE            Divisor to reduce number of entries in
                                    phase histogram.  [x>=0]
    --laser_factor FLOAT RANGE      Factor to correct
                                    dwell_time/laser_frequency.  [x>=0]
    --laser_frequency FLOAT RANGE   Laser frequency in Hz.  [x>=0]
    --pixel_dwell_time FLOAT RANGE  Number of microseconds the scanner
                                    remains at each pixel.  [x>=0]
    --frame_size INTEGER RANGE      Number of pixel in one line scan,
                                    excluding retrace  [x>=0]
    --scanner_line_length INTEGER RANGE
                                    Number of pixels in each line, including
                                    retrace  [x>=0]
    --scanner_line_start INTEGER RANGE
                                    Index of first valid pixel in scan line
                                    [x>=0]
    --scanner_frame_start INTEGER RANGE
                                    Index of first valid pixel after marker.
                                    [x>=0]
    --help                          Show this message and exit.

Example:
    ``python -m lfdfiles.fbd2b64 --laser_factor=1.0041 beads000$EI0T.fbd``

"""

import click
import tifffile

from .lfdfiles import __version__, convert_fbd2b64


def main() -> None:
    """Command line usage main function."""

    @click.command(help='Convert FLIMbox data file to SimFCS B64 files.')
    @click.version_option(version=__version__)
    @click.argument(
        'fbdfile', nargs=1, required=False, type=click.Path(dir_okay=False)
    )
    @click.option(
        '--show',
        default=False,
        is_flag=True,
        show_default=True,
        help='Plot decoded FLIMbox data.',
        type=click.BOOL,
    )
    @click.option(
        '--quiet',
        default=False,
        is_flag=True,
        show_default=True,
        help='Do not print information about file and conversion.',
        type=click.BOOL,
    )
    @click.option(
        '--integrate_frames',
        default=False,
        is_flag=True,
        show_default=True,
        help='Sum all frames.',
        type=click.BOOL,
    )
    @click.option(
        '--full_frame',
        default=False,
        is_flag=True,
        show_default=True,
        help='Return full scanner frames.',
        type=click.BOOL,
    )
    @click.option(
        '--pdiv',
        default=0,
        help='Divisor to reduce number of entries in phase histogram.',
        type=click.IntRange(0),
    )
    @click.option(
        '--laser_factor',
        default=0,
        help='Factor to correct dwell_time/laser_frequency.',
        type=click.FloatRange(0),
    )
    @click.option(
        '--laser_frequency',
        default=0,
        help='Laser frequency in Hz.',
        type=click.FloatRange(0),
    )
    @click.option(
        '--pixel_dwell_time',
        default=0,
        help='Number of microseconds the scanner remains at each pixel.',
        type=click.FloatRange(0),
    )
    @click.option(
        '--frame_size',
        default=0,
        help='Number of pixel in one line scan, excluding retrace',
        type=click.IntRange(0),
    )
    @click.option(
        '--scanner_line_length',
        default=0,
        help='Number of pixels in each line, including retrace',
        type=click.IntRange(0),
    )
    @click.option(
        '--scanner_line_start',
        default=0,
        help='Index of first valid pixel in scan line',
        type=click.IntRange(0),
    )
    @click.option(
        '--scanner_frame_start',
        default=0,
        help='Index of first valid pixel after marker.',
        type=click.IntRange(0),
    )
    def fbd2b64(
        pdiv,
        laser_frequency,
        laser_factor,
        pixel_dwell_time,
        integrate_frames,
        full_frame,
        frame_size,
        scanner_line_length,
        scanner_line_start,
        scanner_frame_start,
        show,
        quiet,
        fbdfile,
    ):
        if not fbdfile:
            fbdfile = tifffile.askopenfilename(
                title='Select a FBD file(s)',
                multiple=False,
                filetypes=[('FBD files', '*.FBD')],
            )
        if fbdfile:
            timer = tifffile.Timer()
            integrate_frames = int(integrate_frames)
            if pdiv <= 0:
                pdiv = -1
            if pixel_dwell_time <= 0:
                pixel_dwell_time = -1
            if laser_factor <= 0:
                laser_factor = -1
            if laser_frequency <= 0:
                laser_frequency = -1
            if frame_size <= 0:
                frame_size = -1
            if scanner_line_length <= 0:
                scanner_line_length = -1
            if scanner_line_start <= 0:
                scanner_line_start = -1
            if scanner_frame_start <= 0:
                scanner_frame_start = -1

            convert_fbd2b64(
                fbdfile,
                show=show,
                verbose=not quiet,
                pdiv=pdiv,
                laser_frequency=laser_frequency,
                laser_factor=laser_factor,
                pixel_dwell_time=pixel_dwell_time,
                square_frame=not full_frame,
                integrate_frames=integrate_frames,
                frame_size=frame_size,
                scanner_line_length=scanner_line_length,
                scanner_line_start=scanner_line_start,
                scanner_frame_start=scanner_frame_start,
            )
            if not quiet and not show:
                print()
                timer.print('Done in')

    fbd2b64(prog_name='fbd2b64')  # pylint: disable=no-value-for-parameter


if __name__ == '__main__':
    main()
