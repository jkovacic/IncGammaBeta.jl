# Copyright (c) 2016, Jernej Kovacic
# Licensed under the MIT "Expat" License.
#
# See LICENSE.md for more details.

#
# Note: code in this file is based on the C++ library "Math", available at:
# https://github.com/jkovacic/math and
# https://github.com/jkovacic/math/blob/master/lib/specfun/CtdFracGeneric.cpp
#


include("settings.jl")
include("exception.jl")


#
# Evaluates the continued fraction:
#
#                        a1
#    f = b0 + -------------------------
#                           a2
#               b1 + -----------------
#                              a3
#                     b2 + ----------
#                           b3 + ...
#
# where ai and bi are functions of 'i'.
#
# Arguments:
# * `fa::Function`: function `a(i)`
# * `fb::Function`: function `b(i)`
#
# The function throws `NoConvergenceError` if the internal
# algorithm does not converge.
#

function ctdfrac_eval(fa::Function, fb::Function)
    #
    # The Lentz's algorithm (modified by I. J. Thompson and A. R. Barnett)
    # is applied to evaluate the continued fraction. The algorithm is
    # presented in detail in:
    #
    #   William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery
    #   Numerical Recipes, The Art of Scientific Computing, 3rd Edition,
    #   Cambridge University Press, 2007
    #
    #   https://books.google.com/books?id=1aAOdzK3FegC&lpg=PA207&ots=3jNoK9Crpj&pg=PA208#v=onepage&f=false
    #
    # The procedure of the algorithm is as follows:
    #
    # - f0 = b0, if b0==0 then f0 = eps
    # - C0 = f0
    # - D0 = 0
    # - for j = 1, 2, 3, ...
    #   -- Dj = bj + aj * D_j-1, if Dj==0 then Dj = eps
    #   -- Cj = bj + aj / C_j-1, if Cj==0 then Cj = eps
    #   -- Dj = 1 / Dj
    #   -- Delta_j = Cj * Dj
    #   -- fj = f_j-1 * Delta_j
    #   -- if abs(Delta_j-1) < TOL then exit for loop
    # - return fj
    #


    # f0 = b0
    f = fb(0)

    const T = typeof(f)
    EPS = eps(T)

    # adjust f0 to eps if necessary
    if abs(f) < EPS
        f = EPS
    end

    # c0 = f0,  d0 = 0
    c = f
    d = T(0)

    # Initially Delta should not be equal to 1
    Delta = T(0)

    j = 1
    while ( abs(Delta-1) > SPECFUN_TOL && j < SPECFUN_MAXITER )
        # obtain 'aj' and 'bj'
        a = fa(j)
        b = fb(j)

        # dj = bj + aj * d_j-1
        d = b + a * d
        # adjust dj to eps if necessary
        if ( abs(d) < EPS )
            d = EPS
        end

        # cj = bj + aj/c_j-1
        c = b + a /c
        # adjust cj to eps if necessary
        if ( abs(c) < EPS )
            c = EPS
        end

        # dj = 1 / dj
        d = T(1) / d

        # Delta_j = cj * dj
        Delta = c * d

        # fj = f_j-1 * Delta_j
        f *= Delta

        # for loop's condition will check, if abs(Delta_j-1)
        # is less than the tolerance

        j += 1
    end  # while

    # check if the algorithm has converged:
    if ( j >= SPECFUN_MAXITER )
        throw(NoConvergenceError)
    end

    # ... if yes, return the fj
    return f;

end
