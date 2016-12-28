# Copyright (c) 2016, Jernej Kovacic
# Licensed under the MIT "Expat" License.
#
# See LICENSE.md for more details.


"""
Required precision for iterative algorithms,
it depends on type 't' that must be derived
from AbstractFloat.
"""
function SPECFUN_ITER_TOL(t)

    if ( Float64 == t )
        retVal = 1e-12
    elseif ( Float32 == t )
        retVal = 1f-6
    else   # t == Float16
        retVal = eps(Float16)
    end

    return retVal
end



"""
Maximum allowed number of iterations for
iterative algorithms.
"""
const SPECFUN_MAXITER = 10000
