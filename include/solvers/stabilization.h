// SPDX-FileCopyrightText: Copyright (c) 2022-2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#ifndef lethe_stabilization_h
#define lethe_stabilization_h

#include <core/vector.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <cmath>

using namespace dealii;

/**
 * @brief Calculate the stabilization parameter for the Navier-Stokes equations
 * in steady-state
 * @return Value of the stabilization parameter - tau
 *
 * @param u_mag Magnitude of the velocity
 * @param kinematic_viscosity Kinematic viscosity
 * @param h Cell size; it should be calculated using the diameter of a sphere of
 * equal volume to that of the cell.
 */
inline double
calculate_navier_stokes_gls_tau_steady(const double u_mag,
                                       const double kinematic_viscosity,
                                       const double h)
{
  return 1. / std::sqrt(Utilities::fixed_power<2>(2. * u_mag / h) +
                        9 * Utilities::fixed_power<2>(4 * kinematic_viscosity /
                                                      (h * h)));
}

/**
 * @brief Calculate the stabilization parameter for the transient Navier-Stokes
 * equations
 *
 * @param u_mag Magnitude of the velocity
 * @param kinematic_viscosity Kinematic viscosity
 * @param h Cell size; it should be calculated using the diameter of a sphere of
 * equal volume to that of the cell.
 * @param sdt Inverse of the time step (1/dt)
 *
 * @return Value of the stabilization parameter - tau
 */

inline double
calculate_navier_stokes_gls_tau_transient(const double u_mag,
                                          const double kinematic_viscosity,
                                          const double h,
                                          const double sdt)
{
  return 1. / std::sqrt(Utilities::fixed_power<2>(sdt) +
                        Utilities::fixed_power<2>(2. * u_mag / h) +
                        9 * Utilities::fixed_power<2>(4 * kinematic_viscosity /
                                                      (h * h)));
}

/**
 * @brief Directional derivative of the GLS parameter with respect to the
 * advective velocity, keeping the stabilization viscosity fixed.
 *
 * @param tau Current stabilization parameter, steady or transient.
 * @param h Cell size used in the stabilization parameter.
 * @param velocity Advective velocity.
 * @param velocity_increment Direction in which velocity is varied.
 * @return Directional derivative of tau.
 */
template <int dim>
inline double
calculate_navier_stokes_gls_tau_velocity_derivative(
  const double          tau,
  const double          h,
  const Tensor<1, dim> &velocity,
  const Tensor<1, dim> &velocity_increment)
{
  // The velocity magnitude used by the assembler is clamped below 1e-12.
  if (velocity.norm() <= 1e-12)
    return 0.;

  return -4. * tau * tau * tau / (h * h) * (velocity * velocity_increment);
}

/**
 * @brief Implements a Moe a posteriori shock capturing method to keep the field bounded. This limiter is only used when DG advection is selected for the Tracer.
 * The limiter is based on the implementation proposed by: Moe, Scott A.,
 * James A. Rossmanith, and David C. Seal. "A simple and effective high-order
 * shock-capturing limiter for discontinuous Galerkin methods." arXiv preprint
 * arXiv:1507.03024 (2015). https://doi.org/10.48550/arXiv.1507.03024
 *
 * The implementation follows the main idea of the paper, but we hardcode the
 * value of alpha in the article to 0. This is supposed to lead to a more
 * diffusive limiter, but in all honesty, all of our tries have shown that
 * this is already a not very dissipative limiter when using BDF time
 * integrator.
 *
 * @param dof_handler The DoFHandler used in the scalar simulation. This DOFHandler must have only a single component (a scalar equation).
 * @param locally_relevant_vector A solution vector that contains the locally_relevant solution. This vector is used to read the solution on the ghost cells and will be modified at the end.
 * @param locally_owned_vector A solution vector that contains the locally_owned solution. This vector will be used to write the new limited solution.
 */
template <int dim>
void
moe_scalar_limiter(const DoFHandler<dim> &dof_handler,
                   GlobalVectorType      &locally_relevant_vector,
                   GlobalVectorType      &locally_owned_vector);
#endif
