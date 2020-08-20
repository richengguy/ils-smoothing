import pathlib

import click
from skimage.io import imread, imsave

from . import ILSSmoothingFilter


@click.command()
@click.option('-y', '--no-input', help='Don\'t ask for any user input.',
              is_flag=True)
@click.argument('input_image', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('output_image', type=click.Path(file_okay=True, dir_okay=False))
def main(no_input: bool, input_image: str, output_image: str):
    '''Real-time Smoothing via Iterative Least Squares'''
    in_path = pathlib.Path(input_image)
    out_path = pathlib.Path(output_image)

    if out_path.exists() and not no_input:
        if click.confirm(f'Overwrite {output_image}?') is False:
            click.echo('Cancelled')
            return

    ils_filter = ILSSmoothingFilter(3.5, 0.8)
    filtered = ils_filter.apply(imread(in_path))
    imsave(out_path, filtered)


if __name__ == '__main__':
    main()
