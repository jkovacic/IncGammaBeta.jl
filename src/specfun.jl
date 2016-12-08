# Copyright (c) 2016, Jernej Kovacic
# Licensed under the MIT "Expat" License.
#
# See LICENSE.md for more details.

#
# Note: code in this file is based on the C++ library "Math", available at:
# https://github.com/jkovacic/math and
# https://github.com/jkovacic/math/blob/master/lib/specfun/SpecFunGeneric.cpp
#


include("ctdfrac.jl")
include("settings.jl")
include("exception.jl")


#
# The following two publications are often referred in this file:
#
# - [Numerical Recipes] or shorter [NR]:
#      William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery
#      Numerical Recipes, The Art of Scientific Computing, 3rd Edition,
#      Cambridge University Press, 2007
#      http://www.nr.com
#
# - [Abramowitz & Stegun] or shorter [AS]:
#      Milton Abramowitz, Irene A. Stegun
#      Handbook of Mathematical Functions with Formulas, Graphs and Mathematical Tables
#      National Bureau of Standards,
#      Applied Mathematics Series 55
#      Issued in June 1964
#
#      The book is copyright free and can be downloaded from:
#      http://www.cs.bham.ac.uk/~aps/research/projects/as/book.php
#



#     betac(x, y)
#
# Unlike Julia 0.5, Julia 0.4 does not support the beta function
# for complex arguments.
# To support regularized beta function on complex numbers as well,
# this convenience function has been introduced.
# TODO: the function may be removed when Julia 0.4 is no longer supported.
function betac{T<:AbstractFloat}(x::Complex{T}, y::Complex{T})

    #
    # It can be shown that B(x,y) can be expressed with gamma
    # functions:
    #
    #             G(x) * G(y)
    #   B(x,y) = -------------
    #               G(x+y)
    #

    const EPS2 = eps(T) * eps(T)
    const gxy = gamma(x + y);

    # handle a very unlikely occassion that G(x,y) gets very close to 0:
    if ( abs2(gxy) < EPS2 )
        throw(SpecfunUndefinedError)
    end

    return gamma(x) * gamma(y) / gxy
end



#     inc_gamma_ctdfrac_closure(a, x)
#
# Return a tupple of parameterized functions `fa(i)` and `fb(i)` that
# evaluate coefficients of the following continued fraction, necessary
# to evaluate the incomplete gamma function:
#
# When x > (1+a), the following continued fraction will be evaluated:
#
#                            1*(1-a)
#   cf = (x-a+1) - ---------------------------
#                                  2*(2-a)
#                    (x-a+3) - ---------------
#                               (x-a+5) - ...
#
# `a(i)` and `b(i)` depend on parameters `a` and `x` and
# are defined as follows:
#
# * a(x,i) = - i * (i-a)
# * b(x,i) = x - a + 1 + 2*i
#
# `a` and `x` are the returned functions' parameters.
#

function inc_gamma_ctdfrac_closure(a, x)
    return i::Integer -> (-i) * (i-a),
           i::Integer -> x - a + 1 + 2*i
end



#     inc_beta_ctdfrac_closure(a, b, x)
#
# Return a tupple of parameterized functions `fa(i)` and `fb(i)` that
# evaluate coefficients of the following continued fraction, necessary
# to evaluate the incomplete beta function:
#
#                     a1
#   cf = 1 + ---------------------
#                       a2
#             1 + ---------------
#                         a3
#                  1 + ---------
#                       1 + ...
# `a(i)` and `b(i)` depend on parameters `a`, `b` and `x` and
# are defined as follows:
#
# * b(x,i) = 1
# *
# *           /    (a+m) * (a+b+m) * x
# *           | - -------------------------   <== i = 2*m+1
# *           /    (a+2*m) * (a + 2*m + 1)
# * a(x,i) = {
# *           \       m * (b-m) * x
# *           |   ---------------------       <== i = 2*m
# *           \    (a+2*m-1) * (a+2*m)
#
# `a`, `b` and `x` are the returned functions' parameters.
#

function inc_beta_ctdfrac_closure(a, b, x)
    T = typeof(x)

    fa =
        function (i::Integer)
            m = div(i, 2)

            return ( 1 == rem(i, 2) ?
                     -((a+m) * (a+b+m) * x) / ((a+i-1) * (a+i)) :
                     (m * (b-m) * x) / ((a+i-1) * (a+i))  )
        end

    fb = i::Integer -> T(1)

    return fa, fb
end




#     incGamma(a, x, upper, reg)
#
# Evaluate an incomplete gamma function with real arguments.
# The exact kind of the returned value depends on parameters
# `upper` and `reg`.
#
# # Arguments:
# * `a`: parameter of the incomplete gamma function
# * `x`: second input argument, the integration limit
# * `upper::Bool`: should the upper (if 'true') or the lower (if 'false') inc. gamma function be returned
# * `reg::Bool`: if 'true', the regularized gamma function is returned, i.e. divided by `gamma(a)`
#
# Note that both `a` and `x` must be strictly greater than 0.
#
# The function throws `SpecfunUndefinedError` when the inc. gamma is
# not defined for any input argument, or `NoConvergenceError`
# when the internal algorithm does not converge.
#
function incGamma{T<:AbstractFloat}(a::T, x::T, upper::Bool, reg::Bool)

    const EPS = eps(T)
    # sanity check:
    if ( a < EPS || x < EPS )
        throw(SpecfunUndefinedError)
    end

    #
    # The algorithm for numerical approximation of the incomplete gamma function
    # as proposed by [Numerical Recipes], section 6.2:
    #
    # When x > (a+1), the upper gamma function can be evaluated as
    #
    #                 -x    a
    #                e   * x
    #   G(a,x) ~= --------------
    #                cf(a,x)
    #
    # where 'cf(a,x) is the continued fraction defined above, its coefficients
    # 'a(i)' and 'b(i)' are implemented in 'inc_gamma_ctdfrac_closure'.
    #
    # When x < (a+1), it is more convenient to apply the following Taylor series
    # that evaluates the lower incomplete gamma function:
    #
    #                          inf
    #                         -----
    #              -x    a    \        G(a)       i
    #   g(a,x) ~= e   * x  *   >    ---------- * x
    #                         /      G(a+1+i)
    #                         -----
    #                          i=0
    #
    # Applying the following property of the gamma function:
    #
    #   G(a+1) = a * G(a)
    #
    # The Taylor series above can be further simplified to:
    #
    #                          inf
    #                         -----              i
    #              -x    a    \                 x
    #   g(a,x) ~= e   * x  *   >    -------------------------
    #                         /      a * (a+1) * ... * (a+i)
    #                         -----
    #                          i=0
    #
    # Once either a lower or an upper incomplete gamma function is evaluated,
    # the other value may be quickly obtained by applying the following
    # property of the incomplete gamma function:
    #
    #   G(a,x) + g(a,x) = G(a)
    #
    # A value of a regularized incomplete gamma function is obtained
    # by dividing g(a,x) or G(a,x) by G(a).
    #


    # This factor is common to both algorithms described above:
    ginc = exp(-x) * (x^a)

    if ( x > (a + 1) )
        #
        # x > (a + 1)
        #
        # In this case evaluate the upper gamma function as described above.
        #

        fa, fb = inc_gamma_ctdfrac_closure(a, x)
        const G = ( true==upper && false==reg ? T(0) : gamma(a) )

        ginc /= ctdfrac_eval(fa, fb)

        #
        # Apply properties of the incomplete gamma function
        # if anything else except a generalized upper incomplete
        # gamma function is desired.
        #
        if ( false == upper )
            ginc = G - ginc
        end

        if ( true == reg )
            # Note: if a>0, gamma(a) is always greater than 0
            ginc /= G
        end
    else
        #
        # x < (a + 1)
        #
        # In this case evaluate the lower gamma function as described above.
        #
        const G = ( false==upper && false==reg ? T(0) : gamma(a) )

        # Initial term of the Taylor series at i=0:
        ginc /= a
        term = ginc

        # Proceed the Taylor series for i = 1, 2, 3... until it converges:
        at = a
        i = 1
        while ( abs(term) > SPECFUN_TOL && i<SPECFUN_MAXITER )
            at += T(1)
            term *= (x / at)
            ginc += term
            i += 1
        end

        # has the series converged?
        if ( i >= SPECFUN_MAXITER )
            throw(NoConvergenceError)
        end

        #
        # Apply properties of the incomplete gamma function
        # if anything else except a generalized lower incomplete
        # gamma function is requested.
        #
        if ( true == upper )
            ginc = G - ginc
        end

        if ( true == reg )
            # Note: if a>0, gamma(a) is always greater than 0
            ginc /= G
        end
    end

    return ginc
end



#     incGamma(a, x, upper, reg)
#
# Evaluate an incomplete gamma function with complex arguments.
# The exact kind of the returned value depends on parameters
# `upper` and `reg`.
#
# # Arguments:
# * `a::Complex`: parameter of the incomplete gamma function
# * `x::Complex`: second input argument, the integration limit
# * `upper::Bool`: should the upper (if 'true') or the lower (if 'false') inc. gamma function be returned
# * `reg::Bool`: if 'true', the regularized gamma function is returned, i.e. divided by `gamma(a)`
#
# Note that unlike at real numbers, incomplete gamma function is defined
# virtually everywhere on the complex plane except at `a` = negative integer.
#
# The function throws `SpecfunUndefinedError` when the inc. gamma is
# not defined for any input argument, or `NoConvergenceError`
# when the internal algorithm does not converge.
#
function incGamma{T<:AbstractFloat}(a::Complex{T}, x::Complex{T}, upper::Bool, reg::Bool)

    #
    # The lower incomplete gamma function can be expanded into:
    #
    #                          inf
    #                         -----              i
    #              -x    a    \                 x
    #   g(a,x) ~= e   * x  *   >    -------------------------
    #                         /      a * (a+1) * ... * (a+i)
    #                         -----
    #                          i=0
    #
    # Once the lower incomplete gamma function is evaluated,
    # the upper one may be quickly obtained by applying the following
    # property of the incomplete gamma function:
    #
    #   G(a,x) + g(a,x) = G(a)
    #
    # A value of a regularized incomplete gamma function is obtained
    # by dividing g(a,x) or G(a,x) by G(a).
    #

    const EPS2 = eps(T) * eps(T)
    # If 'a' equals zero, division by zero would occur
    if ( abs2(a) < EPS2 )
        throw(SpecfunUndefinedError)
    end

    # G = gamma(a) or (0,0) when the value is not used
    const G = ( true==reg || true==upper ? gamma(a) : Complex{T}(0) )

    at = a

    # The first term of the series
    ginc = (x^a) * exp(-x) / a
    term = ginc

    # proceed the series until it converges
    const TOL2 = SPECFUN_TOL * SPECFUN_TOL
    i = 0
    while ( abs2(term) > TOL2 && i < SPECFUN_MAXITER )
        at += T(1)

        # if 'a' is a negative integer, sooner or later this exception will be thrown
        if ( abs2(at) < EPS2 )
            throw(SpecfunUndefinedError)
        end

        term *= x / at
        ginc += term

        i += 1
    end

    # has the series converged?
    if ( i >= SPECFUN_MAXITER )
        throw(NoConvergenceError)
    end

    #
    # Apply properties of the incomplete gamma function
    # if anything else except a non-regularized lower incomplete
    # gamma function is requested.
    #
    if ( true == upper )
        ginc = G - ginc
    end

    if ( true == reg )
        # very unlikely but check it anyway
        if ( abs2(G) < EPS2 )
            throw(SpecfunUndefinedError)
        end

        ginc /= G
    end

    return ginc
end



#     incBeta(a, b, x, lower, reg)
#
# Evaluate the incomplete beta function for real arguments. The exact
# kind of the returned value depends on parameters `upper` and `reg`.
#
# # Arguments
# * `a`: parameter `a` of the beta function
# * `b`: parameter `b` of the beta function
# * `x`: the integration limit
# * `lower::Bool`: should the lower (if 'true') or the upper (if 'false) inc. beta function be returned
# * `reg::Bool`: if 'true', the regularized beta function is returned, i.e. divided by `beta(a, b)`
#
# Note that `a`, `b` and `x` must be strictly greater than 0,
# `x` must be greater than 0 and less than 1
#
# The function throws `SpecfunUndefinedError` when the inc. gamma is
# not defined for any input argument, or `NoConvergenceError`
# when the internal algorithm does not converge.
#

function incBeta{T<:AbstractFloat}(a::T, b::T, x::T, lower::Bool, reg::Bool)

    const EPS = eps(T)

    # sanity check
    if ( a < EPS || b < EPS || x < T(0) || x > T(1) )
        throw(SpecfunUndefinedError)
    end

    #
    # The algorithm is described in detail in [Numerical Recipes], section 6.4
    #
    # If x < (a+1)/(a+b+2), the incomplete beta function will be evaluated as:
    #
    #                  a        b
    #                 x  * (1-x)
    #   Bx(a,b) ~= ------------------
    #                a * cf(x,a,b)
    #
    # where cf(x,a,b) is the continued fraction defined above, its
    # coefficients are implemented in 'inc_beta_ctdfrac_closure'.
    #
    # If x > (a+1)/(a+b+2), the algorithm will converge faster if the following
    # property of the incomplete beta function is applied:
    #
    #   B (a, b) + B   (b, a) = B(a, b)
    #    x          1-x
    #
    # and B_1-x(b,a) can be evaluated as described above.
    #
    # Once either a lower or an upper incomplete beta function is evaluated,
    # the other value may be quickly obtained by applying the following
    # property of the incomplete beta function:
    #
    #   Bx(a,b) + bx(a,b) = B(a,b)
    #
    # A value of a regularized incomplete beta function is obtained
    # by dividing Bx(a,b) or bx(a,b) by B(a,b).
    #

    # checks if x is "large", i.e. greater than the threshold defined above:
    const xlarge = (x > ( (a+T(1)) / (a+b+T(2)) ) )

    # If x is large, the parameters must be swapped
    const xn = ( false==xlarge ? x :  T(1)-x )
    const an = ( false==xlarge ? a : b )
    const bn = ( false==xlarge ? b : a )

    # B = B(a,b) or 0 when the value is not used
    const B = (
        true==reg || (false==xlarge && false==lower) || (true==xlarge  && true==lower) ?
        beta(a, b) : T(0) )

    binc = T(0)

    if ( abs(xn) < EPS )
        # Both boundary conditions are handled separately:
        #   B0(a,b) = 0  and  B1(a,b) = B(a,b)

        binc = ( true==xlarge ? B : T(0) )
    else
        # 'x' is somewhere between 0 and 1, apply the algorithm described above

        fa, fb = inc_beta_ctdfrac_closure(an, bn, xn)
        binc = (xn^an) * ((T(1)-xn)^bn) / an
        binc /= ctdfrac_eval(fa, fb)
    end

    #
    # When x is "large", the algorithm actually returns the
    # upper incomplete beta function!
    #
    # Depending on the requested result ('lower') adjust
    # 'binc' if necessary
    #
    if ( (false==xlarge && false==lower) ||
         (true==xlarge  && true==lower) )
        binc = B - binc
    end

    # Finally regularize the result if requested (via 'reg')
    if ( true == reg )
        # Just in case handle the very unlikely case
        if ( abs(B) < EPS )
            throw(SpecfunUndefinedError)
        end

        binc /= B
    end

    return binc
end


#     incBeta(a, b, x, lower, reg)
#
# Evaluate the incomplete beta function for real arguments. The exact
# kind of the returned value depends on parameters `upper` and `reg`.
#
# # Arguments
# * `a`: parameter `a` of the beta function
# * `b`: parameter `b` of the beta function
# * `x`: the integration limit
# * `lower::Bool`: should the lower (if 'true') or the upper (if 'false) inc. beta function be returned
# * `reg::Bool`: if 'true', the regularized beta function is returned, i.e. divided by `beta(a, b)`
#
# Note that `a`, `b` and `x` must be strictly greater than 0,
# `x` must be greater than 0 and less than 1
#
# The function throws `SpecfunUndefinedError` when the inc. gamma is
# not defined for any input argument, or `NoConvergenceError`
# when the internal algorithm does not converge.
#


#     incBeta(a, b, x, lower, reg)
#
# Evaluate the incomplete beta function for real arguments. The exact
# kind of the returned value depends on parameters `upper` and `reg`.
#
# Arguments
# * `a::Complex`: parameter `a` of the beta function
# * `b::Complex`: parameter `b` of the beta function
# * `x::Complex`: the integration limit
# * `lower::Bool`: should the lower (if 'true') or the upper (if 'false) inc. beta function be returned
# * `reg::Bool`: if 'true', the regularized beta function is returned, i.e. divided by `beta(a, b)`
#
# Note that unlike at real numbers, incomplete beta function is defined
# virtually everywhere except where 'a' is a negative integer, currently this
# function additionally requires that /x/ < 1 otherwise it may not converge.
#
# The function throws `SpecfunUndefinedError` when the inc. gamma is
# not defined for any input argument, or `NoConvergenceError`
# when the internal algorithm does not converge.
#
function incBeta{T<:AbstractFloat}(a::Complex{T}, b::Complex{T}, x::Complex{T}, lower::Bool, reg::Bool)

    #
    # Series expansion of the incomplete beta function as explained at:
    # http://functions.wolfram.com/GammaBetaErf/Beta3/06/01/03/01/01/
    #
    #                a    /                                   2        \
    #               x     |      a*(1-b)*x     a*(1-b)*(2-b)*x         |
    #   Bx(a,b) ~= ---- * | 1 + ----------- + ------------------ + ... | =
    #               a     |        (a+1)           2*(a+2)             |
    #                     \                                            /
    #
    #
    #                     /       inf                                         \
    #                a    |      -----                                     i  |
    #               x     |      \      a * (1-b) * (2-b) * ... * (i-b) * x   |
    #            = ---- * | 1 +   >    -------------------------------------- |
    #               a     |      /                  i! * (a+i)                |
    #                     |      -----                                        |
    #                     \       i=1                                         /
    #
    #

    const EPS2 = eps(T) * eps(T)

    # Division by zero will occur if a==0
    if ( abs2(a) < EPS2 )
        throw(SpecfunUndefinedError)
    end

    # Currently the function requires that /x/ < 1
    # when the series is guaranteed to converge
    if ( abs2(x) >= T(1) )
        throw(SpecfunUndefinedError)
    end

    # B = B(a,b) or (0,0) when the value is not used
    const B = ( true==reg || false==lower ? betac(a, b) : Complex{T}(0) )

    # The first term of the series
    binc = (x^a) / a
    term = binc * a

    at = a;
    bt = -b;

    # proceed the series until it converges
    const TOL2 = SPECFUN_TOL * SPECFUN_TOL
    i = 1
    while ( abs2(term) > TOL2 && i <= SPECFUN_MAXITER )
        at += T(1)
        bt += T(1)

        # if 'a' is a negative integer, sooner or later this exception will be thrown
        if ( abs2(at) < EPS2 )
            throw(SpecfunUndefinedError)
        end

        term *= x * bt / T(i)

        binc += term / at

        i += 1
    end

    # has the series converged?
    if ( i >= SPECFUN_MAXITER )
        throw(NoConvergenceError)
    end

    #
    # Apply properties of the incomplete beta function
    # if anything else except a non-regularized lower incomplete
    # beta function is desired.
    #
    if ( false == lower )
        binc = B - binc
    end

    if ( true == reg )
        # Very unlikely but check it anyway
        if ( abs2(B) < EPS2 )
            throw(SpecfunUndefinedError)
        end

        binc /= B
    end

    return binc
end




"""
    inc_gamma_upper(a, x)

Compute the upper incomplete gamma function, defined as:

           inf
            /
            |  a-1    -t
   G(a,x) = | t    * e   dt
            |
            /
            x

# Arguments:
* `a`: parameter of the incomplete gamma function
* `x`: second input argument, the lower integration limit

For real arguments, `a` and `x` must be greater than 0.
For complex arguments, the function is defined for virtually
any combination of `a` and `x` except when `a` is 0 or a
negative integer.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_gamma_upper(2.0, 0.5)
0.9097960517658588

julia> inc_gamma_upper(2.0, 5.0)
0.040427681994512805

julia> inc_gamma_upper(2.0+im, 1.0-im)
0.294740560104593 + 1.1235453000068967im
```
"""
function inc_gamma_upper(a, x)
    return incGamma(a, x, true, false)
end




"""
    inc_gamma_lower(a, x)

Compute the lower incomplete gamma function, defined as:

            x
            /
            |  a-1    -t
   g(a,x) = | t    * e   dt
            |
            /
            0

# Arguments:
* `a`: parameter of the incomplete gamma function
* `x`: second input argument, the lower integration limit

For real arguments, `a` and `x` must be greater than 0.
For complex arguments, the function is defined for virtually
any combination of `a` and `x` except when `a` is 0 or a
negative integer.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_gamma_lower(3.0, 2.5)
0.9123736299270637

julia> inc_gamma_lower(1.5, 4.0)
0.8454501069992294

julia> inc_gamma_lower(1.4-0.7im, 0.3+0.1im)
-0.016291212634531548 + 0.1305217380308644im
```
"""
function inc_gamma_lower(a, x)
    return incGamma(a, x, false, false)
end




"""
    inc_gamma_upper_reg(a, x)

Compute the regularized upper incomplete gamma function, defined as:

                  inf
                   /
              1    |  a-1    -t
   Q(a,x) = ------ | t    * e   dt
             G(a)  |
                   /
                   x

# Arguments:
* `a`: parameter of the incomplete gamma function
* `x`: second input argument, the lower integration limit

For real arguments, `a` and `x` must be greater than 0.
For complex arguments, the function is defined for virtually
any combination of `a` and `x` except when `a` is 0 or a
negative integer.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_gamma_upper_reg(3.0, 0.5)
0.9856123254685195

julia> inc_gamma_upper_reg(-1.1+0.8im, 2.4-1.3im)
0.001971552504340772 + 0.016252454678230718im
```
"""
function inc_gamma_upper_reg(a, x)
    return incGamma(a, x, true, true)
end




"""
    inc_gamma_lower_reg(a, x)

Compute the regularized lower incomplete gamma function, defined as:

                   x
                   /
              1    |  a-1    -t
   P(a,x) = ------ | t    * e   dt
             G(a)  |
                   /
                   0

# Arguments:
* `a`: parameter of the incomplete gamma function
* `x`: second input argument, the lower integration limit

For real arguments, `a` and `x` must be greater than 0.
For complex arguments, the function is defined for virtually
any combination of `a` and `x` except when `a` is 0 or a
negative integer.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_gamma_lower_reg(1.5, 3.2)
0.9063091948884513

julia> inc_gamma_lower_reg(-3.2-1.4im, -2.0-0.5im)
0.23972825708627568 + 0.4357814237699705im
```
"""
function inc_gamma_lower_reg(a, x)
    return incGamma(a, x, false, true)
end




"""
    inc_beta_lower(a, b, x)

Compute the lower incomplete beta function, defined as:

             x
             /
             |  a-1        b-1
   Bx(a,b) = | t    * (1-t)    dt
             |
             /
             0

# Arguments:
* `a`: parameter `a` of the beta function
* `b`: parameter `b` of the beta function
* `x`: integration limit

For real arguments, `a` and `b` must be greater than 0,
`x` must be greater than 0 and less than 1.
For complex arguments, `a` and `b` can be any complex values except that
`a` must not be 0 or a negative integer. Currently the `x` is restricted
to: /x/ < 1.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_beta_lower(2.0, 5.0, 0.2)
0.011487999926398719

julia> inc_beta_lower(2.0, 5.0, 0.7)
0.032968833333333336

julia> inc_beta_lower(2.0-im, 3.0+im, 0.1+0.1im)
-0.014279981061424203 - 0.010562833267973058im
```
"""
function inc_beta_lower(a, b, x)
    return incBeta(a, b, x, true, false)
end




"""
    inc_beta_upper(a, b, x)

Compute the upper incomplete beta function, defined as:

             1
             /
             |  a-1        b-1
   bx(a,b) = | t    * (1-t)    dt = B(a,b) - Bx(a,b)
             |
             /
             x

# Arguments:
* `a`: parameter `a` of the beta function
* `b`: parameter `b` of the beta function
* `x`: integration limit

For real arguments, `a` and `b` must be greater than 0,
`x` must be greater than 0 and less than 1.
For complex arguments, `a` and `b` can be any complex values except that
`a` must not be 0 or a negative integer. Currently the `x` is restricted
to: /x/ < 1.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_beta_upper(1.0, 2.0, 0.15)
0.36125

julia> inc_beta_upper(1.0, 2.0, 0.82)
0.01620000000000001

julia> inc_beta_upper(-1.2+im, -3.0-im, 0.1-0.2im)
35.01301965887684 + 12.573969351233654im
```
"""
function inc_beta_upper(a, b, x)
    return incBeta(a, b, x, false, false)
end




"""
    inc_beta_lower_reg(a, b, x)

Compute the regularized lower incomplete beta function, defined as:

                      x
                      /
                1     |  a-1        b-1
   Ix(a,b) = -------- | t    * (1-t)    dt
              B(a,b)  |
                      /
                      0

# Arguments:
* `a`: parameter `a` of the beta function
* `b`: parameter `b` of the beta function
* `x`: integration limit

For real arguments, `a` and `b` must be greater than 0,
`x` must be greater than 0 and less than 1.
For complex arguments, `a` and `b` can be any complex values except that
`a` must not be 0 or a negative integer. Currently the `x` is restricted
to: /x/ < 1.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_beta_lower_reg(3.0, 2.0, 0.12)
0.006289919999999998

julia> inc_beta_lower_reg(2.2-0.2im, 1.4+0.7im, -0.3+0.4im)
0.8221304789832569 - 0.40178180311947265imm
```
"""
function inc_beta_lower_reg(a, b, x)
    return incBeta(a, b, x, true, true)
end




"""
    inc_beta_upper_reg(a, b, x)

Compute the regularized upper beta function, defined as:

                      1
                      /
                1     |  a-1        b-1
   ix(a,b) = -------- | t    * (1-t)    dt = 1 - Ix(a,b)
              B(a,b)  |
                      /
                      x

# Arguments:
* `a`: parameter `a` of the beta function
* `b`: parameter `b` of the beta function
* `x`: integration limit

For real arguments, `a` and `b` must be greater than 0,
`x` must be greater than 0 and less than 1.
For complex arguments, `a` and `b` can be any complex values except that
`a` must not be 0 or a negative integer. Currently the `x` is restricted
to: /x/ < 1.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_beta_upper_reg(4.0, 1.0, 0.32)
0.98951424

julia> inc_beta_upper_reg(-3.8-0.7im, -3.4+2.7im, -0.6-0.1im)
0.0010989971419782549 + 0.0021282684016619385im
```
"""
function inc_beta_upper_reg(a, b, x)
    return incBeta(a, b, x, false, true);
end




#     as26_2_22(p)
#
# Implement the formula 26.2.22 in [Abramowitz & Stegun], i.e
# return such `x` that approximately satisfies Q(x) ~= p.
#
# # Arguments:
# * `p` - desired probability
#
# Unlike the original formula, this function accepts `p` being
# greater than 0.5.
#
# This "private" function expects that `p` is greater than 0
# and less than 1.
#
function as26_2_22(p)

    #
    # If 0 < p < 0.5:
    #
    #             +--------+
    #      -+    /      1               +------------+
    #   t =  \  /  ln -----   =    -+  / (-2) * ln(p)
    #         \/       p^2           \/
    #
    # Then initial approximation of 'x' can be calculated as:
    #
    #                 2.30753 + 0.27061 * t
    #   x ~= t - ---------------------------------
    #             1 + t * (0.99229 + 0.04481 * t)
    #
    # If p > 0.5, the expressions above are performed on its complementary
    # value (1-p) and the sign of the final 'x' is reversed.
    #

    T = typeof(p)

    const pp = ( p>= T(0.5) ? T(1) - p : p )

    const t = sqrt(T(-2) * log(pp) )

    # [Abramowitz & Stegun], section 26.2.22:
    const a0 = T(2.30753)
    const a1 = T(0.27061)
    const b0 = T(0.99229)
    const b1 = T(0.04481)

    x = t - (a0 + a1 * t) / (T(1) + t * (b0 + b1 * t))
    if ( p > T(0.5) )
        x = -x
    end

    return x
end


#     invIncGamma(a, g, upper, reg)
#
# Compute the inverse of the incomplete gamma function. The exact kind
# of the returned value depends on parameters `upper` and `reg`.
#
# # Arguments:
# * `a`: parameter of the incomplete gamma function
# * `g`: desired value of the incomplete gamma function
# * `upper::Bool`: if 'true', inverse of the upper incomplete gamma function will be evaluated
# * `reg::Bool`: if 'true', inverse of the regularized incomplete gamma function (divided by `gamma(a)`) will be evaluated
#
# Note that both `a` and `g` must be strictly greater than 0.
#
# The function is defined for real arguments only.
#
# The function throws `SpecfunUndefinedError` when the function is
# not defined for any input argument, or `NoConvergenceError`
# when the internal algorithm does not converge.
#
function invIncGamma{T<:AbstractFloat}(a::T, g::T, upper::Bool, reg::Bool)

    const EPS = eps(T)

    # sanity check
    if ( a < EPS || g < T(0) )
        throw(SpecfunUndefinedError)
    end

    #
    # The algorithm finds an inverse of the regularized lower
    # incomplete gamma function. If 'g' represents a result of
    # any other kind of the incomplete gamma function (specified)
    # by 'upper' and 'reg', it should be converted to a result of
    # the regularized lower incomplete gamma function by applying
    # well known properties of this function.
    #
    # The algorithm is proposed in [Numerical Recipes], section 6.2.1.
    #

    const G = gamma(a)

    p = g;
    if ( false == reg )
        p /= G
    end

    if ( true == upper )
        p = T(1) - p
    end

    # if 'p' equals 0 (or is very close to it),
    # the inverse inc. gamma function will also equal 0.
    if ( abs(p) < EPS )
        return T(0)
    end

    #
    # If 'p' is greater than 1, the incomplete gamma function
    # is not defined at all.
    # If 'p' equals 1 (or is very close to it), the result would
    # be infinity which is not supported by this algorithm.
    #
    if ( p >= (T(1) - EPS) )
        throw(SpecfunUndefinedError)
    end


    #
    # There is no direct series or continued fraction to evaluate
    # an inverse of the incomplete gamma function. On the other hand
    # there are known algorithms that estimate this value very well.
    # This algorithm implements approximation method proposed by the
    # [Numerical Recipes], section 6.2.1 and [Abramowitz & Stegun],
    # sections 26.2.22 and 26.4.17.
    #
    # Note: another method to estimate the initial value of the incomplete
    # gamma function is described in:
    #   Armido R. DiDonato, Alfred H. Morris
    #   Computation of the incomplete gamma function ratios and their inverse
    #   ACM Transactions on Mathematical Software
    #   Volume 12 Issue 4, Dec. 1986, pp. 377-393
    #   http://dl.acm.org/citation.cfm?id=23109&coll=portal&dl=ACM
    #
    # When an approximation is known, the Newton - Raphson method can be
    # applied to refine it further to the desired tolerance.
    #

    x = T(1)

    if ( a <= T(1) )
        #
        # a <= 1:
        #
        # First P(a,1) is approximated by:
        #
        #   Pa = P(a,1) ~= a* (0.253 + 0.12 * a)
        #
        # If 'p' is less than Pa's complement value, 'x' is
        # approximated as:
        #
        #         a      +------+
        #        --+    /   p
        #   x ~=    \  / ------
        #            \/   1-Pa
        #
        # If 'p' is greater than 1-Pa, 'x' is approximated as:
        #
        #                1 - p
        #   x ~= 1 - ln -------
        #                 Pa
        #
        # Note: as long as 0<a<1, Pa will never get even close to 1,
        #       hence the first expression is always defined.
        #

        const c1 = T(0.253)
        const c2 = T(0.12)
        const Pa = a * (c1 + c2 * a)
        const cpa = T(1) - Pa

        x = ( p < cpa ?
                 (p / cpa)^ (T(1) / a) :
                 T(1) - log((T(1) - p)/ Pa)  )
    else
        #
        # a > 1:
        #
        # The initial approximation of 'x' is obtained by the function
        # as26_2_22() that implements [Abramowitz & Stegun], section 26.2.22.
        #
        # Finally the approximated x0 is evaluated as:
        #
        #             /       1          x      \ 3
        #   x0 ~= a * | 1 - ----- + ----------- |
        #             \      9*a     3*sqrt(a)  /
        #

        x = -as26_2_22(p)

        # [Abramowitz & Stegun], section 26.4.17:
        x = T(1) - ( T(1) / (T(9) * a) ) + ( x / ( T(3) * sqrt(a) ) )

        # x = a * x^3:
        x  *= a * x * x
    end


    #
    # When the initial value of 'x' is obtained, a root finding
    # method is applied to determine the exact inverse. This algorithm
    # applies the slightly modified Newton - Raphson method:
    # if 'x' tries to go negative, its half value is assigned
    # to the new value.
    #

    #
    # The Newton - Raphson algorithm also requires the differentiation of
    # the regularized lower incomplete gamma function:
    #
    #                   a-1    -x
    #    d P(a,x)      x    * e
    #   ---------- = --------------
    #       dx         gamma(a)
    #
    # Verified by Maxima:
    # (%i1)  diff(gamma_incomplete_regularized(a, x), x);
    # (%o1)  (x^a-1*%e^-x)/gamma(a)
    #

    xn = T(0)
    f = inc_gamma_lower_reg(a, x) - p
    i = 0
    while ( abs(f) > SPECFUN_TOL && i < SPECFUN_MAXITER )
        xn = x - f * G * exp(x) / (x^(a-T(1)))

        # x must not go negative!
        x = ( xn > EPS ? xn : x * T(0.5) )

        f = inc_gamma_lower_reg(a, x) - p

        i += 1
    end

    # Has the algorithm converged?
    if ( i >= SPECFUN_MAXITER )
        throw(NoConvergenceError)
    end

    return x
end


#     invIncBeta(a, b, y, lower, reg)
#
# Compute the inverse of the incomplete beta function. The exact kind
# of the returned value depends on parameters `lower` and `reg`.
#
# # Arguments:
# * `a`: parameter `a` of the incomplete beta function
# * `b`: parameter `b` of the incomplete beta function
# * `y`: desired value of the incomplete beta function
# * `lower::Bool`: if 'true', inverse of the lower incomplete beta function will be evaluated
# * `reg::Bool`: if 'true', inverse of the regularized incomplete beta function (divided by `beta(a,b)`) will be evaluated
#
# Note that `a`, `b` and `y` must be strictly greater than 0.
#
# The function is defined for real arguments only.
#
# The function throws `SpecfunUndefinedError` when the function is
# not defined for any input argument, or `NoConvergenceError`
# when the internal algorithm does not converge.
#
function invIncBeta{T<:AbstractFloat}(a::T, b::T, y::T, lower::Bool, reg::Bool)

    const EPS = eps(T)

    # sanity check
    if ( a < EPS || b < EPS || y < T(0) )
        throw(SpecfunUndefinedError)
    end

    #
    # The algorithm finds an inverse of the regularized lower
    # incomplete beta function. If 'y' represents a result of
    # any other kind of the incomplete beta function (specified
    # by 'lower' and 'reg'), it should be converted to a result of
    # the regularized lower incomplete beta function by applying
    # well known properties of this function.
    #
    # The algorithm is proposed in [Numerical Recipes], section 6.4.
    #

    const B = beta(a, b)

    p = y

    if ( false == reg )
        p /= B
    end

    if (false == lower)
        p = T(1) - p
    end

    # if 'p' equals 0 (or is very close to it),
    # the inverse inc. beta function will also equal 0.
    if ( abs(p) < EPS )
        return T(0)
    end

    #
    # If 'p' is greater than 1, the incomplete beta function
    # is not defined at all.
    # If 'p' equals 1 (or is very close to it), the result would
    # be infinity which is not supported by this algorithm.
    #
    if ( p >= (T(1) - EPS) )
        throw(SpecfunUndefinedError)
    end

    #
    # There is no known direct series or continued fraction to evaluate
    # an inverse of the incomplete beta function. On the other hand
    # there are known algorithms that estimate this value quite well.
    # This algorithm implements approximation method proposed by the
    # [Numerical Recipes], section 6.4 and [Abramowitz & Stegun],
    # section 26.5.22.
    #
    # When an approximation is known, the Newton - Raphson method can be
    # applied to refine it further to the desired tolerance.
    #

    x = T(0)

    if ( a>=T(1) && b>=T(1) )
        #
        # a >= 1  and  b >= 1:
        #
        # Initial 'x' is approximated as described in [Abramowitz & Stegun],
        # section 26.2.22 (implemented in a separate function).
        #
        # Then it is further refined as described in [Abramowitz & Stegun],
        # section 26.5.22:
        #
        #           1                   1
        #   ta = -------   and  tb = -------
        #         2*a-1               2*b-1
        #
        #              2
        #             x  - 3
        #   lambda = --------
        #               6
        #
        #           2
        #   h = ---------
        #        ta + tb
        #
        #        x * sqrt(h + lambda)                /           5      2   \
        #   w = ---------------------- - (tb - ta) * | lambda + --- - ----- |
        #                h                           \           6     3*h  /
        #
        #                   a
        #   x  ~=   ------------------
        #                       2*w
        #              a + b * e
        #

        # [Abramowitz & Stegun], section 26.2.22:
        x = as26_2_22(p)

        const lambda = (x*x - T(3)) / T(6)
        const ta = T(1) / ( T(2) * a - T(1) )
        const tb = T(1) / ( T(2) * b - T(1) )
        const h = T(2) / (ta + tb)

        const w = x * sqrt(h + lambda) / h - (tb - ta) *
                    (lambda + T(5) / T(6) - T(2) / (T(3) * h) )

        x = a / (a + b * exp(T(2) * w))
    else
        #
        # Either 'a' or 'b' or both are less than 1:
        #
        # In this case the initial 'x' is approximated as
        # described in [Numerical Recipes], section 6.4:
        #
        #
        #         1    /    a    \a              1    /    b    \b
        #   ta = --- * | ------- |    and  tb = --- * | ------- |
        #         a    \  a + b  /               b    \  a + b  /
        #
        #
        #   S = ta + tb
        #
        # If p < ta/S:
        #
        #         a    +----------+
        #   x ~=  -+  / p * S * a
        #           \/
        #
        # If p >= ta/S:
        #
        #        b    +----------+
        #   x ~= -+  / 1 - p*S*b
        #          \/
        #

        const ta = ( (a/(a+b))^a ) / a
        const tb = ( (b/(a+b))^b ) / b
        const S = ta + tb

        x = ( p < ta/S ?   (p*S*a)^(T(1)/a) :  (T(1)-p*S*b)^(T(1)/b) )
    end

    #
    # When the initial value of 'x' is obtained, a root finding
    # method is applied to determine the exact inverse. This algorithm
    # applies the slightly modified Newton - Raphson method:
    # if 'x' tries to go negative or beyond 1, it is bisected between
    # its current value and the upper/lower bondary.
    #

    #
    # The Newton - Raphson algorithm also requires the differentiation of
    # the regularized lower incomplete beta function:
    #
    #                   a-1        b-1
    #    d Ix(a,b)     x    * (1-x)
    #   ----------- = ------------------
    #      dx             beta(a,b)
    #
    # Verified by Maxima:
    # (%i1) diff(beta_incomplete_regularized(a, b, x), x);
    # (%o1) ((1-x)^b-1*x^a-1)/beta(a,b)
    #

    xn = T(0)
    f = inc_beta_lower_reg(a, b, x) - p

    i = 0
    while ( abs(f) > SPECFUN_TOL && i < SPECFUN_MAXITER )
        xn = x - f * B / (x^(a-T(1)) * (T(1)-x)^(b-T(1)) )

        # x must not go negative or beyond 1!
        if ( xn < EPS )
            x *= T(0.5)
        elseif ( xn > (T(1)-EPS) )
            x = (T(1) + x) * T(0.5)
        else
            x = xn
        end

        f = inc_beta_lower_reg(a, b, x) - p

        i += 1
    end

    # Has the algorithm converged?
    if ( i >= SPECFUN_MAXITER )
        throw(NoConvergenceError)
    end

    return x
end





"""
    inc_gamma_lower_inv(a, g)

Compute the inverse of the lower incomplete gamma function,
i.e. return such `x` that satisfies:

    x
    /
    |  a-1    -t
    | t    * e   dt  =  g
    |
    /
    0

# Arguments:
* `a`: parameter of the incomplete gamma function
* `g`: desired value of the lower incomplete gamma function

`a` must be strictly greater than 0, `g` must be greater or equal
to 0 and less than `gamma(a)`.

The function is defined for real arguments only.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_gamma_lower_inv(0.24, 0.94)
0.0020243625046595295

julia> inc_gamma_lower_inv(2.8, 0.17)
0.9873931016522355
```
"""
function inc_gamma_lower_inv(a, g)
    return invIncGamma(a, g, false, false)
end




"""
    inc_gamma_upper_inv(a, g)

Compute the inverse of the upper incomplete gamma function,
i.e. return such `x` that satisfies:

   inf
    /
    |  a-1    -t
    | t    * e   dt  =  g
    |
    /
    x

# Arguments:
* `a`: parameter of the incomplete gamma function
* `g`: desired value of the upper incomplete gamma function

`a` must be strictly greater than 0, `g` must be greater or equal
to 0 and less than `gamma(a)`.

The function is defined for real arguments only.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_gamma_upper_inv(0.65, 0.86)
0.21736643202568437

julia> inc_gamma_upper_inv(3.5, 0.43)
5.609013667436654
```
"""
function inc_gamma_upper_inv(a, g)
    return invIncGamma(a, g, true, false)
end




"""
    inc_gamma_lower_reg_inv(a, g)

Compute the inverse of the regularized lower incomplete gamma function,
i.e. return such `x` that satisfies:

          x
          /
     1    |  a-1    -t
   ------ | t    * e   dt  =  g
    G(a)  |
          /
          0

# Arguments:
* `a`: parameter of the incomplete gamma function
* `g`: desired value of the regularized lower incomplete gamma function

`a` must be strictly greater than 0, `g` must be greater or equal
to 0 and less than 1.

The function is defined for real arguments only.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_gamma_lower_reg_inv(0.2, 0.3)
0.0015877902675184167

julia> inc_gamma_lower_reg_inv(3.0, 0.7)
3.6155679939137437
```
"""
function inc_gamma_lower_reg_inv(a, g)
    return invIncGamma(a, g, false, true)
end




"""
    inc_gamma_upper_reg_inv(a, g)

Compute the inverse of the upper incomplete gamma function,
i.e. return such `x` that satisfies:

         inf
          /
     1    |  a-1    -t
   ------ | t    * e   dt  =  g
    G(a)  |
          /
          x

# Arguments:
* `a`: parameter of the incomplete gamma function
* `g`: desired value of the regularized upper incomplete gamma function

`a` must be strictly greater than 0, `g` must be greater or equal
to 0 and less than 1.

The function is defined for real arguments only.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_gamma_upper_reg_inv(0.3, 0.4)
0.14125250515251847

julia> inc_gamma_upper_reg_inv(5.2, 0.82)
3.129677423009873
```
"""
function inc_gamma_upper_reg_inv(a, g)
    return invIncGamma(a, g, true, true)
end




"""
    inc_beta_lower_inv(a, b, y)

Compute the inverse of the lower incomplete beta function,
i.e. returns such `x` that satisfies:

             x
             /
             |  a-1        b-1
   Bx(a,b) = | t    * (1-t)    dt = y
             |
             /
             0

# Arguments:
* `a`: parameter `a` of the incomplete beta function
* `b`: parameter `b` of the incomplete beta function
* `y`: desired value of the lower incomplete beta function (`Bx(a,b)`)

`a` and `b` must be strictly greater than 0, `y` must be greater or equal
to 0 and less than `beta(a,b)`.

The function is defined for real arguments only.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_beta_lower_inv(2.8, 0.3, 2.0)
0.9997335694367433

julia> inc_beta_lower_inv(1.1, 1.3, 0.4)
0.5190110415182052
```
"""
function inc_beta_lower_inv(a, b, y)
    return invIncBeta(a, b, y, true, false)
end




"""
    inc_beta_upper_inv(a, b, y)

Return the inverse of the upper incomplete beta function,
i.e. it returns such `x` that satisfies:

             1
             /
             |  a-1        b-1
   bx(a,b) = | t    * (1-t)    dt = y
             |
             /
             x

# Arguments:
* `a`: parameter `a` of the incomplete beta function
* `b`: parameter `b` of the incomplete beta function
* `y`: desired value of the upper incomplete beta function (`bx(a,b)`)

`a` and `b` must be strictly greater than 0, `y` must be greater or equal
to 0 and less than `beta(a,b)`.

The function is defined for real arguments only.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_beta_upper_inv(0.4, 0.5, 1.8)
0.41086936325882417

julia> inc_beta_upper_inv(1.7, 1.1, 0.2)
0.720552685909532
```
"""
function inc_beta_upper_inv(a, b, y)
    return invIncBeta(a, b, y, false, false)
end




"""
    inc_beta_lower_reg_inv(a, b, y)

Compute the inverse of the regularized lower incomplete beta function,
i.e. return such `x` that satisfies:

                       x
                       /
                1      |  a-1        b-1
   Ix(a,b) = --------  | t    * (1-t)    dt = y
              B(a,b)   |
                       /
                       0

# Arguments:
* `a`: parameter `a` of the incomplete beta function
* `b`: parameter `b` of the incomplete beta function
* `y`: desired value of the regularized lower incomplete beta function (`Ix(a,b)`)

`a` and `b` must be strictly greater than 0, `y` must be greater or equal
to 0 and less than 1.

The function is defined for real arguments only.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_beta_lower_reg_inv(0.3, 0.2, 0.7)
0.9785534148021483

julia> inc_beta_lower_reg_inv(2.4, 3.5, 0.6)
0.44942826252472956
```
"""
function inc_beta_lower_reg_inv(a, b, y)
    return invIncBeta(a, b, y, true, true)
end




"""
    inc_beta_upper_reg_inv(a, b, y)

Compute the inverse of the regularized upper incomplete beta function,
i.e. return such `x` that satisfies:

                      1
                      /
                1     |  a-1        b-1
   ix(a,b) = -------  | t    * (1-t)    dt = y
              B(a,b)  |
                      /
                      x

# Arguments:
* `a`: parameter `a` of the incomplete beta function
* `b`: parameter `b` of the incomplete beta function
* `y`: desired value of the regularozed upper incomplete beta function (`ix(a,b)`)

`a` and `b` must be strictly greater than 0, `y` must be greater or equal
to 0 and less than 1.

The function is defined for real arguments only.

The function throws `SpecfunUndefinedError` when the function is
not defined for any input argument, or `NoConvergenceError`
when the internal algorithm does not converge.

# Examples:

```jldoctest
julia> inc_beta_upper_reg_inv(0.9, 1.5, 0.7)
0.18163313152662056

julia> inc_beta_upper_reg_inv(1.9, 2.7, 0.25)
0.5648300078293456
```
"""
function inc_beta_upper_reg_inv(a, b, y)
    return invIncBeta(a, b, y, false, true)
end
