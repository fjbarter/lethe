// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#include <solvers/stabilization.h>

#include <../tests/tests.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>

template <int dim>
void
test()
{
  Tensor<1, dim> velocity;
  Tensor<1, dim> increment;
  for (unsigned int d = 0; d < dim; ++d)
    {
      velocity[d]  = 1.2 + 0.7 * d;
      increment[d] = 0.3 - 0.4 * d;
    }

  const double h         = 0.17;
  const double viscosity = 0.02;
  const double epsilon   = 1e-6;
  for (const bool transient : {false, true})
    {
      const auto tau = [&](const Tensor<1, dim> &u) {
        const double speed = std::max(u.norm(), 1e-12);
        return transient ?
                 calculate_navier_stokes_gls_tau_transient(speed,
                                                           viscosity,
                                                           h,
                                                           2.5) :
                 calculate_navier_stokes_gls_tau_steady(speed, viscosity, h);
      };
      const double exact = calculate_navier_stokes_gls_tau_velocity_derivative(
        tau(velocity), h, velocity, increment);
      const double finite_difference = (tau(velocity + epsilon * increment) -
                                        tau(velocity - epsilon * increment)) /
                                       (2. * epsilon);
      AssertThrow(std::abs(exact - finite_difference) < 1e-9,
                  ExcMessage(
                    "GLS velocity derivative differs from finite differences"));
    }

  Tensor<1, dim> zero;
  AssertThrow(std::abs(calculate_navier_stokes_gls_tau_velocity_derivative(
                1., h, zero, increment)) < 1e-30,
              ExcMessage("Clamped velocity must have zero derivative"));
  deallog << dim << "D steady and transient derivatives passed" << std::endl;
}

int
main()
{
  initlog();
  test<2>();
  test<3>();
}
