#!/usr/bin/env julia

#
# A test script that checks correctness of algorithms that evaluate incomplete
# gamma and beta functions and their inverses.
# All input arguments have been carefully selected to cover all branches
# of evaluation algorithms.
#
# For more information how to reproduce exact (i.e. unrounded) results, see
# https://github.com/jkovacic/math/blob/master/scripts/test/specfun.mac and
# https://github.com/jkovacic/math/blob/master/scripts/test/specfun.py
#


using IncGammaBeta
using Base.Test


# Round all results to 'i' decimal places:
function my_round(i::Integer)
    return x -> round(x, i)
end

mrd = my_round(6)


print_with_color(:blue, "Running tests...\n")

@test mrd( inc_gamma_upper(2.0, 0.5) ) == 0.909796
@test mrd( inc_gamma_upper(2.0, 5.0) ) == 0.040428
@test mrd( inc_gamma_lower(3.0, 2.5) ) == 0.912374
@test mrd( inc_gamma_lower(1.5, 4.0) ) == 0.84545
@test mrd( inc_gamma_upper_reg(3.0, 0.5) ) == 0.985612
@test mrd( inc_gamma_lower_reg(1.5, 3.2) ) == 0.906309
@test mrd( inc_gamma_upper(2.0+im, 1.0-im) ) == 0.294741 + 1.123545im
@test mrd( inc_gamma_lower(1.4-0.7im, 0.3+0.1im) ) == -0.016291 + 0.130522im
@test mrd( inc_gamma_upper_reg(-1.1+0.8im, 2.4-1.3im) ) == 0.001972 + 0.016252im
@test mrd( inc_gamma_lower_reg(-3.2-1.4im, -2.0-0.5im) ) == 0.239728 + 0.435781im

@test mrd( inc_beta_lower(2.0, 5.0, 0.2) ) == 0.011488
@test mrd( inc_beta_lower(2.0, 5.0, 0.7) ) == 0.032969
@test mrd( inc_beta_upper(1.0, 2.0, 0.15) ) == 0.36125
@test mrd( inc_beta_upper(1.0, 2.0, 0.82) ) == 0.0162
@test mrd( inc_beta_lower_reg(3.0, 2.0, 0.12) ) == 0.00629
@test mrd( inc_beta_upper_reg(4.0, 1.0, 0.32) ) == 0.989514
@test mrd( inc_beta_lower(2.0-im, 3.0+im, 0.1+0.1im) ) == -0.01428 - 0.010563im
@test mrd( inc_beta_upper(-1.2+im, -3.0-im, 0.1-0.2im) ) == 35.01302 + 12.573969im
@test mrd( inc_beta_lower_reg(2.2-0.2im, 1.4+0.7im, -0.3+0.4im) ) ==  0.82213 - 0.401782im
@test mrd( inc_beta_upper_reg(-3.8-0.7im, -3.4+2.7im, -0.6-0.1im) ) == 0.001099 + 0.002128im

@test mrd( inc_gamma_lower_reg_inv(0.2, 0.3) ) == 0.001588
@test mrd( inc_gamma_lower_reg_inv(3.0, 0.7) ) == 3.615568
@test mrd( inc_gamma_upper_reg_inv(0.3, 0.4) ) == 0.141253
@test mrd( inc_gamma_upper_reg_inv(5.2, 0.82) ) == 3.129677
@test mrd( inc_gamma_lower_inv(0.24, 0.94) ) == 0.002024
@test mrd( inc_gamma_lower_inv(2.8, 0.17) ) == 0.987393
@test mrd( inc_gamma_upper_inv(0.65, 0.86) ) == 0.217366
@test mrd( inc_gamma_upper_inv(3.5, 0.43) ) == 5.609014

@test mrd( inc_beta_lower_reg_inv(0.3, 0.2, 0.7) ) == 0.978553
@test mrd( inc_beta_lower_reg_inv(2.4, 3.5, 0.6) ) == 0.449428
@test mrd( inc_beta_upper_reg_inv(0.9, 1.5, 0.7) ) == 0.181633
@test mrd( inc_beta_upper_reg_inv(1.9, 2.7, 0.25) ) == 0.56483
@test mrd( inc_beta_lower_inv(2.8, 0.3, 2.0) ) == 0.999734
@test mrd( inc_beta_lower_inv(1.1, 1.3, 0.4) ) == 0.519011
@test mrd( inc_beta_upper_inv(0.4, 0.5, 1.8) ) == 0.410869
@test mrd( inc_beta_upper_inv(1.7, 1.1, 0.2) ) == 0.720553


print_with_color(:blue, "Tests completed successfully.\n")
