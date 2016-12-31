# Copyright (c) 2016, Jernej Kovacic
# Licensed under the MIT "Expat" License.
#
# See LICENSE.md for more details.


module IncGammaBeta

export inc_gamma_upper, inc_gamma_lower, inc_gamma_upper_reg, inc_gamma_lower_reg
export inc_gamma_lower_inv, inc_gamma_upper_inv, inc_gamma_lower_reg_inv, inc_gamma_upper_reg_inv
export inc_beta_lower, inc_beta_upper, inc_beta_lower_reg, inc_beta_upper_reg
export inc_beta_lower_inv, inc_beta_upper_inv, inc_beta_lower_reg_inv, inc_beta_upper_reg_inv
export SpecfunUndefinedError, NoConvergenceError

include("specfun.jl")
include("exception.jl")



"""
A Julia package with functions that implement incomplete gamma and beta functions
of all kinds (upper and lower, regularized and unegularized). All functions except
inverse ones also support complex arguments.

The package is written entirely in Julia and does not require any 3rd party libraries
or additional Julia packages.
"""
IncGammaBeta


end  # module
