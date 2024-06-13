#include "NonLinearOpt.h"
#include <fstream>

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>


#define DIM 2

using namespace dealii;

class ElasticBar
{
public:
  ElasticBar();
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results();
  void apply_boundary_conditions();

  Triangulation<DIM> triangulation;
  FESystem<DIM> fe;
  DoFHandler<DIM> dof_handler;
  
  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> solution;
  Vector<double> system_rhs;

  // MappingQ1<DIM> mapping;  // Add this line

};

ElasticBar::ElasticBar()
  : fe(FE_Q<DIM>(1), DIM), dof_handler(triangulation)//, mapping()
{}

void ElasticBar::make_grid()
{
  GridGenerator::hyper_rectangle(triangulation, Point<DIM>(0.0,0.0), Point<DIM>(5.0,1.0), true);
  triangulation.refine_global(1);
}

void ElasticBar::setup_system()
{
  dof_handler.distribute_dofs(fe);
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // Define the mapping
  // mapping = MappingQ1<DIM, DIM>();  

  output_results();
}

void ElasticBar::assemble_system()
{
  QGauss<DIM> quadrature_formula(fe.degree + 1);
  FEValues<DIM> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
        {
          cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                fe_values.shape_grad(j, q_index) *
                                fe_values.JxW(q_index));
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    for (const unsigned int i : fe_values.dof_indices())
    {
      for (const unsigned int j : fe_values.dof_indices())
      {
        system_matrix.add(local_dof_indices[i],
                          local_dof_indices[j],
                          cell_matrix(i, j));
      }
    }
  }

    apply_boundary_conditions();

}

void ElasticBar::apply_boundary_conditions()
{
  std::map<types::global_dof_index,double> boundary_values;
  
  
  // Apply a specified displacement for the x-direction at boundary 1
  std::vector<bool> mask(DIM, false);
  mask[0] = true;  // Only consider x-direction
  ComponentMask component_mask(mask);

  // Apply zero displacement for the x-direction at boundary 0
  VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<DIM>(), boundary_values, component_mask);
  VectorTools::interpolate_boundary_values(dof_handler, 1, ConstantFunction<DIM>(0.1), boundary_values, component_mask);

  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

// void ElasticBar::solve()
// {
//   SparseDirectUMFPACK A                        _direct;
//   A_direct.initialize(system_matrix);
//   A_direct.vmult(solution, system_rhs);
// }

// write the solver using conjugate gradient technique
void ElasticBar::solve()
{
  SolverControl solver_control(1000, 1e-12);
  SolverCG<> solver(solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  // std::cout << "   system_matrix = " << system_matrix << std::endl;

  solver.solve(system_matrix, solution, system_rhs, preconditioner); 
  std::cout << "   system_rhs = " << system_rhs.size() << ", " << system_rhs << std::endl;
  std::cout << "   solution = " << solution.size() << ", " << solution << std::endl;

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;
}


void ElasticBar::run()
{
  make_grid();
  setup_system();

  assemble_system();
  solve();
  output_results();
}

void ElasticBar::output_results()
{
  // if (DIM == 3)
  // {
  //   DataOut<3> data_out;
  //   data_out.attach_dof_handler(dof_handler);
  //   std::vector<std::string> solution_names = {"displacement_x", "displacement_y", "displacement_z"};
  //   data_out.add_data_vector(solution, solution_names);
  //   data_out.build_patches();
  //   std::ofstream output("solution.vtk");
  //   data_out.write_vtk(output);

  // }

    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
        std::vector<std::string> solution_names = {"displacement_x", "displacement_y"};
    data_out.add_data_vector(solution, solution_names);
    data_out.build_patches();
    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);
}

int main ()
{
  ElasticBar elastic_bar;
  elastic_bar.run();
  return 0;
}

// int main ()
// {
//   compressed_strip::ElasticProblem ep;

//   char fileName[MAXLINE];
//   std::cout << "Please enter an input file: " << std::endl;
//   std::cin >> fileName;
//   ep.read_input_file(fileName);

//   ep.create_mesh();

//   ep.setup_system();

//   std::cout << "\n   Number of active cells:       "
//             << ep.get_number_active_cells()
//             << std::endl;


//   std::cout << "   Number of degrees of freedom: "
//             << ep.get_n_dofs()
//             << std::endl << std::endl;

//   ep.solve_forward_problem();
//   exit(0);

//   return(0);
// }
