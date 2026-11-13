// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/**
 * @brief Compare the outlet Jacobian with finite differences of its RHS.
 * A large tangential velocity exposes the coupling to normal velocity during
 * backflow. Geometry is fixed and only the outlet assembler is evaluated.
 */
#include <core/simulation_control.h>

#include <solvers/copy_data.h>
#include <solvers/navier_stokes_assemblers.h>
#include <solvers/navier_stokes_scratch_data.h>
#include <solvers/simulation_parameters.h>

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>

#include <../tests/tests.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

/**
 * @brief Check every outlet Jacobian column on a fixed cell.
 * @param degree Finite-element polynomial degree.
 * @param normal_velocity Outward normal velocity; negative means backflow.
 * @param beta Backflow penalty coefficient.
 */
template <int dim>
void
check_jacobian(const unsigned int degree,
               const double       normal_velocity,
               const double       beta)
{
  SimulationParameters<dim>     parameters;
  ParameterHandler              handler;
  Parameters::SizeOfSubsections sizes;
  sizes.boundary_conditions = 1;
  parameters.declare(handler, sizes);
  parameters.parse(handler);
  const auto control =
    std::make_shared<SimulationControlTransient>(parameters.simulation_control);
  const FESystem<dim>          fe(FE_Q<dim>(degree), dim, FE_Q<dim>(degree), 1);
  const MappingQ1<dim>         mapping;
  const QGauss<dim>            quadrature(degree + 2);
  const QGauss<dim - 1>        face_quadrature(degree + 2);
  NavierStokesScratchData<dim> scratch(control,
                                       parameters.physical_properties_manager,
                                       fe,
                                       quadrature,
                                       mapping,
                                       face_quadrature);
  Triangulation<dim>           triangulation;
  const double                 cell_width = 0.005;
  GridGenerator::hyper_cube(triangulation, 0., cell_width, true);
  DoFHandler<dim> dofs(triangulation);
  dofs.distribute_dofs(fe);
  const unsigned int face   = 2; // y = 0, with outward normal -y.
  const unsigned int n_dofs = fe.n_dofs_per_cell();
  scratch.fe_face_values.reinit(dofs.begin_active(), face);

  // Populate only the face data used by the outlet assembler, excluding all
  // volume physics, particle geometry and time integration from this check.
  scratch.n_faces          = GeometryInfo<dim>::faces_per_cell;
  scratch.n_faces_q_points = face_quadrature.size();
  scratch.is_boundary_cell = true;
  scratch.is_boundary_face.assign(scratch.n_faces, false);
  scratch.is_boundary_face[face] = true;
  scratch.boundary_face_id.resize(scratch.n_faces);
  scratch.boundary_face_id[face] = 2;
  scratch.cell_size              = cell_width;
  scratch.face_JxW.reinit(scratch.n_faces, scratch.n_faces_q_points);
  scratch.face_normal.reinit(scratch.n_faces, scratch.n_faces_q_points);
  scratch.face_phi_u.reinit(scratch.n_faces, scratch.n_faces_q_points, n_dofs);
  scratch.face_velocity_values.assign(
    scratch.n_faces, std::vector<Tensor<1, dim>>(scratch.n_faces_q_points));
  Tensor<1, dim> base_velocity;
  base_velocity[1]                = -normal_velocity;
  base_velocity[dim == 3 ? 2 : 0] = 5000.;
  for (unsigned int q = 0; q < scratch.n_faces_q_points; ++q)
    {
      scratch.face_JxW[face][q]    = scratch.fe_face_values.JxW(q);
      scratch.face_normal[face][q] = scratch.fe_face_values.normal_vector(q);
      scratch.face_velocity_values[face][q] = base_velocity;
      for (unsigned int j = 0; j < n_dofs; ++j)
        scratch.face_phi_u[face][q][j] =
          scratch.fe_face_values[FEValuesExtractors::Vector(0)].value(j, q);
    }
  BoundaryConditions::NSBoundaryConditions<dim> boundaries;
  boundaries.type[2] = BoundaryConditions::BoundaryType::outlet;
  boundaries.beta[2] = beta;
  OutletBoundaryCondition<dim>         assembler(control, boundaries);
  StabilizedMethodsTensorCopyData<dim> matrix_data(n_dofs, quadrature.size());
  assembler.assemble_matrix(scratch, matrix_data);

  if (normal_velocity == 0.)
    {
      // min(0,u.n) is not differentiable at zero. The assembler selects the
      // inactive branch; a central difference here is not a valid reference.
      assembler.assemble_rhs(scratch, matrix_data);
      AssertThrow(matrix_data.local_matrix.frobenius_norm() < 1e-14 &&
                    matrix_data.local_rhs.l2_norm() < 1e-14,
                  ExcMessage(
                    "The outlet must be inactive at zero normal flow"));
    }
  else
    for (const double epsilon : {1e-5})
      {
        FullMatrix<double> finite_difference(n_dofs, n_dofs);
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            StabilizedMethodsTensorCopyData<dim> plus(n_dofs,
                                                      quadrature.size());
            StabilizedMethodsTensorCopyData<dim> minus(n_dofs,
                                                       quadrature.size());
            for (unsigned int q = 0; q < scratch.n_faces_q_points; ++q)
              scratch.face_velocity_values[face][q] =
                base_velocity + epsilon * scratch.face_phi_u[face][q][j];
            assembler.assemble_rhs(scratch, plus);
            for (unsigned int q = 0; q < scratch.n_faces_q_points; ++q)
              scratch.face_velocity_values[face][q] =
                base_velocity - epsilon * scratch.face_phi_u[face][q][j];
            assembler.assemble_rhs(scratch, minus);
            for (unsigned int i = 0; i < n_dofs; ++i)
              finite_difference(i, j) =
                -(plus.local_rhs(i) - minus.local_rhs(i)) / (2. * epsilon);
          }
        FullMatrix<double> error(matrix_data.local_matrix);
        error.add(-1., finite_difference);
        const double scale = std::max(1., finite_difference.frobenius_norm());
        AssertThrow(error.frobenius_norm() / scale < 1e-7,
                    ExcMessage("Outlet Jacobian does not match the residual derivative"));
      }
  deallog << dim << "D Q" << degree << " "
          << (beta == 0.           ? "disabled" :
              normal_velocity < 0. ? "backflow" :
              normal_velocity > 0. ? "outflow" :
                                     "zero normal flow")
          << ": OK" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  for (const unsigned int degree : {1, 2})
    for (const double normal : {-10., -1e-4, 0., 1e-4, 10.})
      {
        check_jacobian<2>(degree, normal, 1.);
        check_jacobian<3>(degree, normal, 1.);
      }
  check_jacobian<2>(1, -10., 0.);
}
