import pathlib

import click
import numpy as np
from skimage.io import imread, imsave

from . import ILSSmoothingFilter


@click.command()
@click.option('-y', '--no-input', help='Don\'t ask for any user input.',
              is_flag=True)
@click.option('-s', '--smoothing', default=3.5, show_default=True,
              help='Smoothing amount.')
@click.option('-e', '--edge-preservation', default=0.8, show_default=True,
              help='Edge preservation/retention amount.')
@click.option('-n', '--iterations', default=4, show_default=True,
              help='Number of filtering iterations.')
@click.argument('input_image', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('output_image', type=click.Path(file_okay=True, dir_okay=False))
def main(no_input: bool, smoothing: float, edge_preservation: float,
         iterations: int, input_image: str, output_image: str):
    '''Real-time Smoothing via Iterative Least Squares

    The ILS filter is an edge-aware filter that can selectively blur weak edges
    while retaining strong ones.  The `-s` and `-e` options will control the
    filter.
    '''
    in_path = pathlib.Path(input_image)
    out_path = pathlib.Path(output_image)

    if out_path.exists() and not no_input:
        if click.confirm(f'Overwrite {output_image}?') is False:
            click.echo('Cancelled')
            return

    # Load in the image.
    img = imread(in_path)
    out = np.zeros_like(img)

    # Filter each channel independently.
    ils = ILSSmoothingFilter(smoothing, edge_preservation, iterations)
    for i in range(img.ndim):
        out[:, :, i] = ils.apply(img[:, :, i])

    # Save the result; JPEG has the quality set to 100 since the default is 75.
    args = {}
    if out_path.suffix != '.png':
        args['quality'] = 100
    imsave(out_path, out, **args)


if __name__ == '__main__':
    main()
