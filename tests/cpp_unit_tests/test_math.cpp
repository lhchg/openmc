#include <boost/math/distributions/students_t.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <random>
#include <cmath>

#include "openmc/math_functions.h"
#include "openmc/random_dist.h"
#include "openmc/wmp.h"

TEST_CASE("Test t_percentile") {
    // Permutations include 1 DoF, 2 DoF, and > 2 DoF
    // We will test 5 p-values at 3-DoF values
    std::vector<double> test_ps = {0.02, 0.4, 0.5, 0.6, 0.98};
    std::vector<int> test_dfs = {1, 2, 5};

    // The reference solutions come from boost::math
    for (int df : test_dfs) {
        std::vector<double> ref_ts;
        std::vector<double> test_ts;

        for (double p : test_ps) {
            double ref_t = boost::math::quantile(boost::math::students_t(df), p);
            double test_t = openmc::t_percentile(p, df);

            ref_ts.push_back(ref_t);
            test_ts.push_back(test_t);
        }

        // The 5 DoF approximation in openmc.lib.math.t_percentile is off by up to
        // 8e-3 from the scipy solution, so test that one separately with looser
        // tolerance
        double tolerance = (df > 2) ? 1e-2 : 1e-6;

        for (size_t i = 0; i < test_ps.size(); ++i) {
            REQUIRE(std::abs(ref_ts[i] - test_ts[i]) < tolerance);
        }
    }
}


TEST_CASE("Test calc_pn") {
    const int max_order = 10;
    std::vector<double> test_xs = { -1.0, -0.5, 0.0, 0.5, 1.0 };

    std::vector<std::vector<double>> test_vals;
    std::vector<std::vector<double>> ref_vals;
    for (double x : test_xs) {
        // Reference solutions from boost::math
        std::vector<double> ref_val;

        for (int n = 0; n <= max_order; ++n) {
            ref_val.push_back(boost::math::legendre_p(n, x));
        }
        ref_vals.push_back(ref_val);

        std::vector<double> test_val(max_order+1);
        openmc::calc_pn_c(max_order, x, test_val.data());
        test_vals.push_back(test_val);
        
    }

    for (int i = 0;  i < ref_vals.size(); i++) {
        REQUIRE_THAT(ref_vals[0], Catch::Matchers::Approx(test_vals[0]));
    }
}


TEST_CASE("Test evaluate_legendre") {
    const int max_order = 10;
    std::vector<double> test_xs = { -1.0, -0.5, 0.0, 0.5, 1.0 };

    std::vector<double> ref_vals;

    for (double x : test_xs) {
        double sum = 0.0;

        for (int l = 0; l <= max_order; ++l) {
            // Coefficients are set to 1, but will incorporate the (2l+1)/2 norm factor
            // for the reference solution
            double coeff = 0.5 * (2 * l + 1);
            sum += coeff * boost::math::legendre_p(l, x);
        }

        ref_vals.push_back(sum);
    }


    // Set the coefficients back to 1s for the test values since
    // evaluate legendre incorporates the (2l+1)/2 term on its own
    std::vector<double> test_coeffs(max_order+1, 1.0); 

    std::vector<double> test_vals;
    for (double x : test_xs) {
        test_vals.push_back(openmc::evaluate_legendre(test_coeffs.size() - 1, test_coeffs.data(), x));
    }

    REQUIRE_THAT(ref_vals, Catch::Matchers::Approx(test_vals));
}


double coeff(int n, int m) {
    return std::sqrt((2.0 * n + 1) * boost::math::factorial<double>(n - m) /
                      boost::math::factorial<double>(n + m));
}

double pnm_bar(int n, int m, double mu) {
    double val = coeff(n, m);
    if (m != 0) {
        val *= std::sqrt(2.0);
    }
    val *= boost::math::legendre_p(n, m, mu);
    return val;
}

TEST_CASE("Test calc_rn") {
    const int max_order = 10;

    const double azi = 0.1; //Longitude
    const double pol = 0.2; //Latitude
    const double mu = std::cos(pol);

    // Reference solutions from the equations
    std::vector<double> ref_vals;

    for (int n = 0; n <= max_order; ++n) {
        for (int m = -n; m <= n; ++m) {
            double ylm;
            if (m < 0) {
                ylm = pnm_bar(n, std::abs(m), mu) * std::sin(std::abs(m) * azi);
            } else {
                ylm = pnm_bar(n, m, mu) * std::cos(m * azi);
            }

            // Un-normalize for comparison
            ylm /= std::sqrt(2.0 * n + 1);
            ref_vals.push_back(ylm);
        }
    }    

    std::vector<double> test_uvw {std::sin(pol) * std::cos(azi),
                                  std::sin(pol) * std::sin(azi),
                                  std::cos(pol)};

    std::vector<double> test_vals((max_order + 1) * (max_order + 1), 0);
    openmc::calc_rn_c(max_order, test_uvw.data(), test_vals.data());

    REQUIRE_THAT(ref_vals, Catch::Matchers::Approx(test_vals));
}

TEST_CASE("Test calc_zn") {
    const int n = 10;
    const double rho = 0.5; 
    const double phi = 0.5;

    std::vector<double> ref_vals{
        1.00000000e+00, 2.39712769e-01, 4.38791281e-01,
        2.10367746e-01, -5.00000000e-01, 1.35075576e-01,
        1.24686873e-01, -2.99640962e-01, -5.48489101e-01,
        8.84215021e-03, 5.68310892e-02, -4.20735492e-01,
        -1.25000000e-01, -2.70151153e-01, -2.60091773e-02,
        1.87022545e-02, -3.42888902e-01, 1.49820481e-01,
        2.74244551e-01, -2.43159131e-02, -2.50357380e-02,
        2.20500013e-03, -1.98908812e-01, 4.07587508e-01,
        4.37500000e-01, 2.61708929e-01, 9.10321205e-02,
        -1.54686328e-02, -2.74049397e-03, -7.94845816e-02,
        4.75368705e-01, 7.11647284e-02, 1.30266162e-01,
        3.37106977e-02, 1.06401886e-01, -7.31606787e-03,
        -2.95625975e-03, -1.10250006e-02, 3.55194307e-01,
        -1.44627826e-01, -2.89062500e-01, -9.28644588e-02,
        -1.62557358e-01, 7.73431638e-02, -2.55329539e-03,
        -1.90923851e-03, 1.57578403e-02, 1.72995854e-01,
        -3.66267690e-01, -1.81657333e-01, -3.32521518e-01,
        -2.59738162e-02, -2.31580576e-01, 4.20673902e-02,
        -4.11710546e-04, -9.36449487e-04, 1.92156884e-02,
        2.82515641e-02, -3.90713738e-01, -1.69280296e-01,
        -8.98437500e-02, -1.08693628e-01, 1.78813094e-01,
        -1.98191857e-01, 1.65964201e-02, 2.77013853e-04 };

    int nums = ((n + 1) * (n + 2)) / 2;

    std::vector<double> test_vals(nums, 0);
    openmc::calc_zn(n, rho, phi, test_vals.data());

    REQUIRE_THAT(ref_vals, Catch::Matchers::Approx(test_vals));
}

TEST_CASE("Test calc_zn_rad") {
    const int n = 10;
    const double rho = 0.5;

    std::vector<double> ref_vals {
        1.00000000e+00, -5.00000000e-01, -1.25000000e-01,
        4.37500000e-01, -2.89062500e-01,-8.98437500e-02};

    int nums =  n / 2 + 1;
    std::vector<double> test_vals(nums, 0);
    openmc::calc_zn_rad(n, rho, test_vals.data());

    REQUIRE_THAT(ref_vals, Catch::Matchers::Approx(test_vals));
}

TEST_CASE("Test rotate_angle") {
    std::vector<double> uvw0{1.0, 0.0, 0.0};
    const double phi = 0.0;
    double mu = 0.0;

    std::vector<double> ref_uvw {0.0, 0.0, -1.0};

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dist(0, 1);

    uint64_t prn_seed = 0;
    for (int i = 0; i < 63; ++i) {
        prn_seed |= static_cast<std::uint64_t>(dist(gen)) << i;
    }

    std::vector<double> test_uvw = uvw0;
    openmc::rotate_angle_c(test_uvw.data(), mu, &phi, &prn_seed);

    REQUIRE_THAT(ref_uvw, Catch::Matchers::Approx(test_uvw));

    ref_uvw = {1.0, 0.0, 0.0};
    mu = 1.0;
    test_uvw = uvw0;
    openmc::rotate_angle_c(test_uvw.data(), mu, &phi, &prn_seed);

    REQUIRE_THAT(ref_uvw, Catch::Matchers::Approx(test_uvw));


    ref_uvw = {0.9, -0.422746750548505, 0.10623175090659095};
    mu = 0.9;
    prn_seed = 1;
    test_uvw = uvw0;
    openmc::rotate_angle_c(test_uvw.data(), mu, NULL, &prn_seed);

    REQUIRE_THAT(ref_uvw, Catch::Matchers::Approx(test_uvw));
}

TEST_CASE("Test maxwell_spectrum") {
    const double T = 0.5;
    const double ref_val = 0.27767406743161277;

    uint64_t prn_seed = 1;

    double test_val = openmc::maxwell_spectrum(T, &prn_seed);

    REQUIRE(ref_val == test_val);
}

TEST_CASE("Test watt_spectrum") {
    const double a = 0.5;
    const double b = 0.75;

    uint64_t prn_seed = 1;

    double ref_val = 0.30957476387766697;

    double test_val = openmc::watt_spectrum(a, b, &prn_seed);

    REQUIRE(ref_val == test_val);
}

#include <iterator>
#include <fstream>

TEST_CASE("Test normal_dist") {
    const double mean = 14.08;
    double stdev = 0.0;

    uint64_t prn_seed = 1;

    double ref_val = 14.08;

    double test_val = openmc::normal_variate(mean, stdev, &prn_seed);
    REQUIRE(ref_val == test_val);


    stdev = 1.0;
    const double num_samples = 10000;

    std::vector<double> samples;

    for (int i = 0; i < num_samples; i++) {
        samples.push_back(openmc::normal_variate(mean, stdev, &prn_seed));
        prn_seed += 1;
    }

    std::ofstream file("output.txt");
    std::ostream_iterator<double> out_iter(file, ", ");
    std::copy(samples.begin(), samples.end(), out_iter);
}

TEST_CASE("Test broaden_wmp_polynomials") {
    const double test_E = 0.5;
    double test_dopp = 100.0;  // approximately U235 at room temperature
    const int n = 6;

    std::vector<double> ref_val {2., 1.41421356, 1.0001, 0.70731891, 0.50030001, 0.353907};

    std::vector<double> test_val(n, 0);
    openmc::broaden_wmp_polynomials(test_E, test_dopp, n, test_val.data());

    REQUIRE_THAT(ref_val, Catch::Matchers::Approx(test_val));

    test_dopp = 5.0;
    ref_val = {1.99999885, 1.41421356, 1.04, 0.79195959, 0.6224, 0.50346003};
    openmc::broaden_wmp_polynomials(test_E, test_dopp, n, test_val.data());

    REQUIRE_THAT(ref_val, Catch::Matchers::Approx(test_val));
}

