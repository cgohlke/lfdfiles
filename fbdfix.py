# fbdfix.py

"""Remove data containing non-frame-markers from FLIMbox FBD files.

Usage: python fbdfix.py directory_or_file_name

"""

import os

import lfdfiles
import numpy


def fbdfix(
    fbdfile,
    outfile=None,
    nmarkers=3,
    frame_duration=None,
    frame_duration_maxdiff=None,
):
    """Remove data containing non-frame-markers from FLIMbox FBD file."""
    if outfile is None:
        dirname, fname = os.path.split(fbdfile)
        outfile = os.path.join(dirname, '_' + fname)

    with lfdfiles.FlimboxFbd(fbdfile) as fbd:
        bins, times, markers = fbd.decode()

    frame_durations = numpy.diff(times[markers]).astype(numpy.int64)
    if len(frame_durations) <= nmarkers:
        return 'skipped'

    if frame_duration is None:
        # assume longest of last few frames is good
        frame_duration = numpy.max(frame_durations[-nmarkers:])
    if frame_duration_maxdiff is None:
        frame_duration_maxdiff = frame_duration // 512

    n = nmarkers
    i = len(frame_durations)
    while i > 0:
        i -= 1
        if abs(frame_durations[i] - frame_duration) > frame_duration_maxdiff:
            n -= 1
            if n == 0:
                break
        else:
            n = nmarkers
    else:
        i = -nmarkers

    i += nmarkers
    skip_bytes = max(0, markers[i] * 2 - 4)

    with open(fbdfile, 'rb') as fh:
        header = fh.read(32 * 1024)
        fh.seek(skip_bytes, 1)
        data = fh.read()

    with open(outfile, 'wb') as fh:
        fh.write(header)
        fh.write(data)

    return f'removed {skip_bytes} bytes, {i} of {markers.size} markers'


def main():
    """Command line script main function."""
    import glob
    import sys

    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if '*' in arg:
            files = glob.glob(arg, recursive=True)
        elif os.path.isdir(arg):
            files = glob.glob(os.path.join(arg, '**/*.fbd'), recursive=True)
        else:
            files = [arg]
    else:
        files = []

    if not files:
        print(__doc__)
        return

    if len(files) > 1:
        common = len(os.path.commonpath(files))
        if files[0][common] == os.path.sep:
            common += 1
        if common > 0:
            print(files[0][:common])
    else:
        common = 0

    for filename in files:
        fname = os.path.split(filename)[-1]
        if fname.startswith('_') or not fname.lower().endswith('.fbd'):
            continue
        print()
        print('*', filename[common:])
        try:
            print(' ', fbdfix(filename))
        except Exception as exc:
            print(f'  failed: {exc.__class__.__name__}: {exc}')


if __name__ == '__main__':
    main()

# mypy: allow-untyped-defs, allow-untyped-calls
