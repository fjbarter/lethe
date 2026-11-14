// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/**
 * Check the documented normal lubrication force, its cutoff and its units.
 */
#include <core/ib_particle.h>
#include <core/parameters.h>

#include <fem-dem/ib_particles_dem.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <../tests/fem-dem/ib_particles_dem_test_utilities.h>
#include <../tests/tests.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace dealii;

template <int dim>
void
check_force(const double scale, const double requested_gap, const bool periodic)
{
  const double r1 = 0.04 * scale;
  const double r2 = 0.06 * scale;
  const double hmin = 0.001 * scale;
  const double hmax = 0.05 * scale;
  const double gap = requested_gap * scale;
  const double viscosity = 0.01 / scale;
  const double speed = scale;
  auto parameters = make_ib_particles_parameters<dim>(
    dim == 2 ? "0; 0" : "0; 0; 0");

  Point<dim> a;
  Point<dim> b;
  for (unsigned int d = 0; d < dim; ++d)
    a[d] = b[d] = 0.5 * scale;
  a[0] = periodic ? 0.05 * scale : 0.4 * scale;
  b[0] = periodic ? a[0] + scale - (r1 + r2 + gap) :
                    a[0] + r1 + r2 + gap;
  auto first = make_ib_sphere<dim>(0, r1, a);
  auto second = make_ib_sphere<dim>(1, r2, b);
  second.velocity[0] = periodic ? speed : -speed;
  second.velocity[1] = 0.5 * speed;
  first.initialize_previous_solution();
  second.initialize_previous_solution();
  IBParticlesDEM<dim> pair;
  pair.initialize(parameters, nullptr, nullptr, MPI_COMM_WORLD, {first, second});
  if (periodic)
    {
      Tensor<1, dim> offset;
      offset[0] = scale;
      pair.set_periodic_offset(offset);
    }
  pair.update_contact_candidates();
  std::vector<Tensor<1, 3>> forces(2), torques(2);
  pair.calculate_pp_lubrication_force(0., hmax, hmin, viscosity, forces, torques);
  const double effective_diameter = (2. * r1 * 2. * r2) / (2. * r1 + 2. * r2);
  const double expected = gap < hmax ?
    1.5 * numbers::PI * viscosity * effective_diameter * effective_diameter *
      (1. / std::max(gap, hmin) - 1. / hmax) * speed : 0.;
  const double direction = periodic ? 1. : -1.;
  AssertThrow(std::abs(forces[0][0] - direction * expected) <
                1e-10 * std::max(scale, std::abs(expected)),
              ExcMessage("Incorrect particle-particle lubrication force"));
  AssertThrow((forces[0] + forces[1]).norm() < 1e-10 * scale &&
                std::abs(forces[0][1]) < 1e-10 * scale,
              ExcMessage("Lubrication must be normal and conserve momentum"));

  // Test the wall limit on x=0; keep all other walls outside the search range.
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., scale, true);
  triangulation.refine_global(3);
  const FESystem<dim> fe(FE_Q<dim>(1), dim + 1);
  DoFHandler<dim> dofs(triangulation);
  dofs.distribute_dofs(fe);
  Point<dim> position;
  for (unsigned int d = 0; d < dim; ++d)
    position[d] = 0.5 * scale;
  position[0] = r1 + gap;
  auto particle = make_ib_sphere<dim>(0, r1, position);
  particle.velocity[0] = -speed;
  particle.velocity[1] = 0.5 * speed;
  particle.initialize_previous_solution();
  std::vector<IBParticle<dim>> particles{particle};
  IBParticlesDEM<dim> wall;
  wall.initialize(parameters, nullptr, nullptr, MPI_COMM_WORLD, particles);
  wall.update_particles_boundary_contact(particles, dofs, QGauss<dim - 1>(2), MappingQ1<dim>());
  std::vector<Tensor<1, 3>> wall_force(1), wall_torque(1);
  wall.calculate_pw_lubrication_force(0., hmax, hmin, viscosity, wall_force, wall_torque);
  const double diameter = 2. * r1;
  const double expected_wall = gap < hmax ?
    1.5 * numbers::PI * viscosity * diameter * diameter *
      (1. / std::max(gap, hmin) - 1. / hmax) * speed : 0.;
  AssertThrow(std::abs(wall_force[0][0] - expected_wall) <
                1e-10 * std::max(scale, std::abs(expected_wall)) &&
                std::abs(wall_force[0][1]) < 1e-10 * scale,
              ExcMessage("Incorrect particle-wall lubrication force"));
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  for (const double scale : {1., 1000.})
    for (const double gap : {0.0005, 0.01, 0.06})
      {
        check_force<2>(scale, gap, false);
        check_force<3>(scale, gap, false);
        check_force<2>(scale, gap, true);
        check_force<3>(scale, gap, true);
      }
  deallog << "Lubrication force magnitude, direction, cutoff, periodicity and units: OK" << std::endl;
}
