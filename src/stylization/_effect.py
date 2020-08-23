import abc

import click
import numpy as np


class Effect(abc.ABC):
    '''Definition of an artistic effect.

    The interface requires two methods to be implemented: :meth:`_validate`
    :meth:`apply`.  If :meth:`_validate` succeed (no exceptions) then the image
    can be passed along to :meth:`apply`.  This allow for multiple effects to be
    chained together.
    '''
    def can_process(self, img: np.ndarray, *, show_reason: bool = True) -> bool:
        '''Indicate if the effect can process this type of image.

        Parameters
        ----------
        img : np.ndarray
            input image
        show_reason : bool, optional
            if ``True`` then it will print an error to stderr if the image
            cannot be processed

        Returns
        -------
        bool
            ``True`` if the effect will be able to process the image
        '''
        try:
            self._validate(img)
            return True
        except ValueError as e:
            if show_reason:
                click.secho('Error: ', bold=True, fg='red', nl=False, err=True)
                click.echo(str(e), err=True)
            return False

    @abc.abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        '''Apply the effect onto an image.

        The effect should be able to process any image when its implementation
        of :meth:`can_process` returns ``True``.  Otherwise it should raise a
        ValueError

        Parameters
        ----------
        img : np.ndarray
            input image

        Returns
        -------
        np.ndarray
            processed image

        Raises
        ------
        ValueError
            if the image type cannot be processed
        '''

    @abc.abstractmethod
    def _validate(self, img: np.ndarray):
        '''Check to see if an image can be processed.

        Parameters
        ----------
        img : np.ndarray
            image to validate

        Raises
        ------
        ValueError
            with the reason why the image cannot be processed
        '''
