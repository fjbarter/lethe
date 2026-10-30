// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

// Exercise the search on a flat coarse mesh and after mesh/DoF changes. The
// two-rank run also checks ghost cells and rejects artificial-cell interiors.
#include <core/lethe_grid_tools.h>

#include <fem-dem/sharp_ib_stencil_search.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <../tests/tests.h>

#include <map>
#include <set>
#include <vector>

using namespace dealii;

/**
 * @brief Check local containment, cached queries, and missing remote cells.
 * @param dofs Current mesh and DoFs.
 * @param search Initialized search object.
 */
template <int dim>
void
check_queries(const DoFHandler<dim> &dofs, SharpIBStencilSearch<dim> &search)
{
  using Cell = typename DoFHandler<dim>::active_cell_iterator;
  std::map<unsigned int, std::set<Cell>> neighbors;
  LetheGridTools::vertices_cell_mapping(dofs, neighbors);
  Cell seed = dofs.begin_active();
  while (!seed->is_locally_owned())
    ++seed;
  for (const auto &cell : dofs.active_cell_iterators())
    {
      const auto point = cell->center();
      const auto found = search.find(point, seed, neighbors);
      if (cell->is_artificial())
        AssertThrow(found == dofs.end(),
                    ExcMessage("Artificial cell supplied a stencil"));
      else
        {
          AssertThrow(found == cell, ExcMessage("Wrong cell at its center"));
          AssertThrow(search.find(point, seed, neighbors) == found,
                      ExcMessage("Cached query changed the result"));
          // A face may belong to either adjacent cell, but must remain a valid
          // and deterministic choice when the same query is repeated.
          const auto face_point = cell->face(0)->center();
          const auto face_cell  = search.find(face_point, seed, neighbors);
          AssertThrow(face_cell != dofs.end() &&
                        GeometryInfo<dim>::distance_to_unit_cell(
                          MappingQ1<dim>().transform_real_to_unit_cell(
                            face_cell, face_point)) < 1e-12,
                      ExcMessage("Shared-face containment failed"));
          AssertThrow(search.find(face_point, seed, neighbors) == face_cell,
                      ExcMessage("Shared-face cache is inconsistent"));
        }
    }
  Point<dim> outside;
  outside[0] = -10.;
  AssertThrow(search.find(outside, seed, neighbors) == dofs.end(),
              ExcMessage("Exterior point accepted"));
  AssertThrow(search.find(outside, seed, neighbors) == dofs.end(),
              ExcMessage("Cached exterior point accepted"));
}

/** @brief Exercise both flat and refined meshes in the given dimension. */
template <int dim>
void
test()
{
  parallel::distributed::Triangulation<dim> mesh(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(mesh, 6);
  DoFHandler<dim> dofs(mesh);
  FE_Q<dim>       fe(1);
  dofs.distribute_dofs(fe);
  SharpIBStencilSearch<dim> search;
  search.reinit(dofs);
  check_queries(dofs, search);

  // A mesh signal must discard cached misses as well as successful queries.
  Tensor<1, dim> displacement;
  displacement[0] = 0.25;
  GridTools::shift(displacement, mesh);
  check_queries(dofs, search);

  mesh.refine_global(1);
  dofs.distribute_dofs(fe);
  search.reinit(dofs);
  check_queries(dofs, search);

  // DoF redistribution need not change the triangulation. Callers explicitly
  // reinitialize the search in this case.
  dofs.distribute_dofs(fe);
  search.reinit(dofs);
  check_queries(dofs, search);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << dim << "D: containment, cache, ghost cells and invalidation OK"
            << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test<2>();
  test<3>();
}
