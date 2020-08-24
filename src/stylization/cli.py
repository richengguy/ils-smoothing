import pathlib

import click

from . import rotoscope, toning
from ._types import CommonOptions


@click.group()
@click.option('-y', '--no-input', help='Don\'t ask for any user input.',
              is_flag=True)
@click.argument('input_image', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('output_image', type=click.Path(file_okay=True, dir_okay=False))
@click.pass_context
def main(ctx: click.Context, no_input: bool, input_image: str, output_image: str):
    '''Apply a stylization effect to an image.

    The 'COMMAND' is one of the operations below.  They all have default options
    but can be adjusted as needed.  Use 'COMMAND --help' for more details.
    '''
    options = CommonOptions(
        input_image=pathlib.Path(input_image),
        output_image=pathlib.Path(output_image)
    )

    if options.output_image.exists() and not no_input:
        if click.confirm(f'Overwrite {output_image}?') is False:
            click.echo('Cancelled')
            return

    ctx.obj = options


main.add_command(rotoscope.command)
main.add_command(toning.command)
