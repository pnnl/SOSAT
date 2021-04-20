from abc import ABC, abstractmethod


class StressConstraint(ABC):
    """
    Abstract base class for all stress constraint classes.


    """
    @abstractmethod
    def likelihood(self,
                   ss):
        """
        Abstract method that requires all derived classes to implement
        a funcction that computes the likelihood of the constraint for
        a the stress states passed in through the StressState parameter

        Parameters
        ----------
        ss: `SOSAT.StressState` object
            StressState object containing the stress states
            over which the likelihood is to be evaluated

        Returns
        -------
        Numpy MaskedArray
            The returned object is a Numpy MaskedArray containing the
            likelihood for each stress `ss`. The returned array is
            masked identically to `ss.shmin_grid`
        """
        pass
