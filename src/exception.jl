# Copyright (c) 2016, Jernej Kovacic
# Licensed under the MIT "Expat" License.
#
# See LICENSE.md for more details.

"""
Exception thrown when any input argument is not valid.
"""
type SpecfunUndefinedError <: Exception end




"""
Exception thrown when the internal algorithm does not converge.
"""
type NoConvergenceError <: Exception end
