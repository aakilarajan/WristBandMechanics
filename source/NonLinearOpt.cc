/* ---------------------------------------------------------------------
 *
 *
 *
 *
 *
 * ---------------------------------------------------------------------
 *
 *
 * Author: Andrew Akerson
 */

#ifndef COMPRESSEDSTRIPPACABLOCH_CC_
#define COMPRESSEDSTRIPPACABLOCH_CC_
#include "NonLinearOpt.h"

#include <deal.II/grid/grid_tools.h>
#include <fstream>
#include <iostream>
#include <string>
#include <functional>

#include <sys/types.h>
#include <sys/stat.h>


#define _USE_MATH_DEFINES

#define MU_VALUE 1.0
#define NU_VALUE 0.0

#define DIM 3
#define DIM_IP 2
#define DIM3 3
#define STEP_OUT 10
#define PI 3.14159



namespace compressed_strip
{
  using namespace dealii;

  double f_function(double current_t, double maxLoad)
  {

    if(current_t < 0.5)
      return 2.0*maxLoad*current_t;
    else if(current_t < 1.0)
      return -2.0*maxLoad*(current_t - 0.5) + maxLoad;
    else
      return 0.0;
  }

  double load_function_norm(Point<DIM> &p)
  {
//    return -(-fabs(p(0)- 15.0)/30.0 + 0.5);
    double root_var = 0.08;

    if(fabs(p(0)) < 0.15)
      return -1.0/(root_var*sqrt(2.0*M_PI))*exp(-0.5*(p(0))*(p(0))/(root_var*root_var));
    else
      return 0.0;

  }


  inline
  void transform_stress(Tensor<2,DIM> &P, Point<DIM> &q_point, Tensor<2,DIM> &out)
  {
    double theta = atan2(q_point(1),q_point(0));
    Tensor<2,DIM> Rot;
    Rot[0][0] = cos(theta);
    Rot[0][1] = sin(theta);
    Rot[1][0] = -sin(theta);
    Rot[1][1] = cos(theta);
    Rot[2][2] = 1.0;


    out = transpose(Rot)*P*Rot;
  }


  // this will give an error for DIM != 3
  // inline
  // double compute_von_misses_stress(Tensor<2,DIM> &cauchy_stress)
  // {
  //   double von_misses = 0.0;

  //   von_misses += 0.5*(cauchy_stress[0][0] - cauchy_stress[1][1])*(cauchy_stress[0][0] - cauchy_stress[1][1]);
  //   von_misses += 0.5*(cauchy_stress[1][1] - cauchy_stress[2][2])*(cauchy_stress[1][1] - cauchy_stress[2][2]);
  //   von_misses += 0.5*(cauchy_stress[2][2] - cauchy_stress[0][0])*(cauchy_stress[2][2] - cauchy_stress[0][0]);

  //   von_misses += 3.0*(cauchy_stress[0][1] * cauchy_stress[0][1] + 
  //                      cauchy_stress[1][2] * cauchy_stress[1][2] + 
  //                      cauchy_stress[2][0] * cauchy_stress[2][0]);

  //   return von_misses;

  // }
  inline
  double compute_von_misses_stress(Tensor<2,DIM> &cauchy_stress)
  {
    double von_misses = 0.0;

    von_misses += 0.5*(cauchy_stress[0][0] - cauchy_stress[1][1])*(cauchy_stress[0][0] - cauchy_stress[1][1]);
    von_misses += 0.5*(cauchy_stress[1][1])*(cauchy_stress[1][1]);
    von_misses += 0.5*(- cauchy_stress[0][0])*(- cauchy_stress[0][0]);

    von_misses += 3.0*(cauchy_stress[0][1] * cauchy_stress[0][1]);

    return von_misses;

  }


  // computes right hand side values if we were to have body forces. But it just
  // always returns zeros because we don't.
  void ElasticProblem::right_hand_side (const std::vector<Point<DIM> > &points,
                        std::vector<Tensor<1, DIM> >   &values)
  {
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));
    Assert (DIM >= 2, ExcNotImplemented());

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
    {
      values[point_n][0] = 0.0;
      values[point_n][1] = 0.0;
    }

  }

  inline 
  void ElasticProblem::get_gradu_tensor(std::vector<Tensor<1,DIM>> &old_solution_gradient, 
                                          Tensor<2,DIM> &gradu)
  {
    
    gradu = 0.0;
    for (unsigned int i = 0; i < DIM; i ++)
    {
      for(unsigned int j = 0; j < DIM; j++)
      {
        gradu[i][j] += old_solution_gradient[i][j];
      }
    }
  }

  inline void ElasticProblem::get_strain(std::vector<Tensor<1,DIM> > &old_solution_gradient,
      Tensor<2,DIM> &Eps)
  {
    Eps = 0.0;
    for (unsigned int i = 0; i < DIM; i ++)
      for(unsigned int j = 0; j < DIM; j++)
        Eps[i][j] += 0.5*(old_solution_gradient[i][j] + old_solution_gradient[j][i]);
  }


  ElasticProblem::ElasticProblem ()
    :
    dof_handler (triangulation),
    fe (FE_Q<DIM>(1), DIM)  {
  }


  ElasticProblem::~ElasticProblem ()
  {
    dof_handler.clear ();
    // delete postprocess;
  }


  void ElasticProblem::create_mesh()
  {

    Point<DIM> corner1, corner2;
    for (unsigned int i = 0; i<DIM ;  i++)
      corner1[i] = 0.0;

    for (unsigned int i = 0; i<DIM ;  i++)
      corner2[i] = domain_dimensions[i];

    GridGenerator::subdivided_hyper_rectangle(triangulation, grid_dimensions, corner1, corner2, true);
 

  }

  void ElasticProblem::setup_system (const bool initial_step)
  {

    if (initial_step)
    {

    current_time = 0.0;

    N = triangulation.n_active_cells();
      dof_handler.distribute_dofs (fe);

    elmMats.resize(N);
    mu = Shear_mod;
    nu = nu_poi_ratio;
    // for (unsigned int i = 0; i < N; ++i)
    //   elmMats[i].init(mu, nu);

    std::cout << "  mu and nu = " << mu << ", " << nu << std::endl;

    double nq_points = qy*qx*qz;
    dof_handler.distribute_dofs (fe);
    present_solution.reinit (dof_handler.n_dofs());
    solution_u.reinit (dof_handler.n_dofs());
    // solution_u1.reinit (dof_handler.n_dofs());
    
    present_solution = 0.0;
    // solution_u1 = 0.0;
    solution_u = 0.0;
    setup_system_constraints();

    }

    newton_update.reinit (dof_handler.n_dofs());
    newton_update = 0.0;

    system_rhs.reinit (dof_handler.n_dofs());


    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);

    const unsigned int  number_dofs = dof_handler.n_dofs();

    // std::vector<Point<DIM>> support_points(dof_handler.n_dofs());
    // MappingQ1<DIM> mapping;
    // DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    sparsity_pattern.copy_from (dsp);


//    GridTools::distort_random(0.4, triangulation, true);
    // mass_matrix.reinit(sparsity_pattern);
    system_matrix.reinit (sparsity_pattern);

    // get the dofs that we will apply dirichlet condition to
    homo_dofs.resize(dof_handler.n_dofs(), false);


    QGauss<1> quad_x(qx);
    QGauss<1> quad_y(qy);
    QGauss<1> quad_z(qz);

    problemQuadrature = QAnisotropic<DIM>(quad_x, quad_y, quad_z);



    if (initial_step)
    {
    // setting up the phase field
    // still need to filter it yet. just checking for nw
    phi.reinit(N);
    phi_filtered.reinit(N);
    assign_phi();
    
    // setting up von misses stress
    von_misses_stress.reinit(N);
    // initialize all cauchy stress vectors to N size vector
    cauchy1.reinit(N);
    cauchy2.reinit(N);
    cauchy3.reinit(N);
    cauchy12.reinit(N);
    cauchy13.reinit(N);
    cauchy23.reinit(N);

    // initialize fatigue field values
    se_goodman.reinit(N);
    cycles_to_failure.reinit(N);

    assign_material_properties();
      output_results(0);

    }

  }

  void ElasticProblem::compute_fatigue_data()
  {
    // loop over cells and put them in their groups
    typename DoFHandler<DIM>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    
    for (; cell!=endc; ++cell)
    {
      
      unsigned int cell_index = cell->active_cell_index();
      double stress_for_computing_fatigue = cauchy1[cell_index];

      if (stress_for_computing_fatigue >= 2.0 * su_ultimate_strength)
      {
        // std::cout << "XXXXXXXXXXX         MATERIAL FAILURE DETECTED AT CELL " << cell_index << "         XXXXXXXXXXX" << std::endl;
        se_goodman[cell_index] = su_ultimate_strength;
        cycles_to_failure[cell_index] = 0.0;
      }
      else
      {
        se_goodman[cell_index] = stress_for_computing_fatigue/(1.0 - stress_for_computing_fatigue/(2.0 * su_ultimate_strength));
        cycles_to_failure[cell_index] = 0.5 * pow( (se_goodman[cell_index] /303.9), 28.9);
      }

      if (cycles_to_failure[cell_index] <= 1000.0)
        cycles_to_failure[cell_index] = 1000.0;
      else if (cycles_to_failure[cell_index] > 1.0e8)
        cycles_to_failure[cell_index] = 1.0e8;
        
    }
  }


  void ElasticProblem::compute_objective()
  {
        typename DoFHandler<DIM>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    
    for (; cell!=endc; ++cell)
    {
      
      unsigned int cell_index = cell->active_cell_index();
      if (cycles_to_failure[cell_index] >= 1000.0 && cycles_to_failure[cell_index] <= 1.0e8)
        objective += 1.0/ cycles_to_failure[cell_index];
        
    }
  }

  // void ElasticProblem::compute_objective()
  // {
  //       // loop over cells and put them in their groups
  //   typename DoFHandler<DIM>::active_cell_iterator
  //   cell = dof_handler.begin_active(),
  //   endc = dof_handler.end();

  //   	FEValues<DIM> fe_values(fe, problemQuadrature,
	// 		update_values | update_quadrature_points | update_JxW_values);


  // 	unsigned int n_q_points = problemQuadrature.size();

    

  //   for (; cell!=endc; ++cell)
  //   {
  //     fe_values.reinit(cell);
	// 			unsigned int cell_index = cell->active_cell_index();
				
	// 			for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
	// 			{
	// 				for (unsigned int i = 0; i < DIM; i++)
	// 				{
  //           objective += von_misses_stress[cell_index] * fe_values.JxW(q_point);
	// 				}
	// 			}


  //   }

  // }

  void ElasticProblem::assign_phi()
  {

    // loop over cells and put them in their groups
    typename DoFHandler<DIM>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    Point<DIM> current_cell_center;
    cu_thiccness = 10.0e-6;
    // cu_thiccness = domain_dimensions[2]/5.0;
    cu_width = 3000.0e-6;
    double w_freq = 10.0;

    double phi_min = 0.01;
    double phi_substrate = 0.5;
    double phi_electrode = 1.0;

    // double phi_min = 1.0;
    // double phi_substrate = 1.0;
    // double phi_electrode = 1.0;
    

    for (; cell!=endc; ++cell)
    {

      unsigned int cell0_index = cell->active_cell_index();
      current_cell_center = cell->center();

      phi[cell0_index] = phi_substrate;

      // if (current_cell_center[2] < domain_dimensions[2] - cu_thiccness)
      //   phi[cell0_index] = phi_substrate;
      // else if (current_cell_center[1] > domain_dimensions[1]/2.0 - domain_dimensions[1]/8.0 && current_cell_center[1] < domain_dimensions[1]/2.0 + domain_dimensions[1]/8.0)
      //   phi[cell0_index] = phi_electrode;
      // else
      //   phi[cell0_index] = phi_min;

      if (current_cell_center[2] > domain_dimensions[2] - cu_thiccness)
      {
        phi[cell0_index] = phi_min;
        // double y_max = domain_dimensions[1]/2.0 + sin(2.0 * current_cell_center[0] * 2.0 * PI/(domain_dimensions[0]))*domain_dimensions[1]/6.0 + domain_dimensions[1]/8.0;
        // double y_min = domain_dimensions[1]/2.0 + sin(2.0 * current_cell_center[0] * 2.0 * PI/(domain_dimensions[0]))*domain_dimensions[1]/6.0 - domain_dimensions[1]/8.0;
        double y_max = domain_dimensions[1]/2.0 + sin(w_freq * current_cell_center[0] * 2.0 * PI/(domain_dimensions[0]))*domain_dimensions[1]/6.0 + cu_width;
        double y_min = domain_dimensions[1]/2.0 + sin(w_freq * current_cell_center[0] * 2.0 * PI/(domain_dimensions[0]))*domain_dimensions[1]/6.0 - cu_width;
        
        if (current_cell_center[1] > y_min && current_cell_center[1] < y_max)
          phi[cell0_index] = phi_electrode;
      }

      // serpentine - not exactly, but close to serpentine.
      
      // if (current_cell_center[2] > domain_dimensions[2] - cu_thiccness)
      // {
      //   phi[cell0_index] = phi_min;
      //   double center_line  = 0.0;
      //   double serp_radius = 225.0e-6;
      //   double serp_dia = 2.0 * serp_radius;

      //   double x_prime = current_cell_center[0] - ( ) ;

      //   double y_max = domain_dimensions[1]/2.0 + sin(w_freq * current_cell_center[0] * 2.0 * PI/(domain_dimensions[0]))*domain_dimensions[1]/6.0 + cu_width;
      //   double y_min = domain_dimensions[1]/2.0 + sin(w_freq * current_cell_center[0] * 2.0 * PI/(domain_dimensions[0]))*domain_dimensions[1]/6.0 - cu_width;
      // }


    }
    
  }

  



  void ElasticProblem::assign_material_properties()
  {
    // loop over cells and put them in their groups
    typename DoFHandler<DIM>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    
    for (; cell!=endc; ++cell)
    {
      
      unsigned int cell0_index = cell->active_cell_index();
      elmMats[cell0_index].init(phi[cell0_index]*mu, nu);

      if (phi[cell0_index] >= 0.9 && phi[cell0_index] < 1.1)
        elmMats[cell0_index].init(mu_cu, nu_cu);
    }
  }


  void ElasticProblem::setup_system_constraints ()
  {

    constraints.clear ();

    constraints.close ();

    // now do hanging nodes. Because some of the constraints might refer to the same dof
    // for both the symmetry constraint and the hanging node constraint, we will make them
    // separate, then merge them, giving precedence to the hanging node constraints;

//    ConstraintMatrix hanging_node_constraints;
    AffineConstraints<double> hanging_node_constraints;
    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints (dof_handler, hanging_node_constraints);
    hanging_node_constraints.close();

    constraints.merge(hanging_node_constraints, AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
  }


  // void ElasticProblem::apply_boundaries_to_rhs(Vector<double> *rhs, std::vector<bool> *homogenous_dirichlet_dofs)
  // {
  //   for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
  //   {
  //     if ((*homogenous_dirichlet_dofs)[i] == true)
  //       (*rhs)[i] = 0.0;
  //   }
  // }

  // void ElasticProblem::initiate_guess()
  // {
  //   std::vector<bool> side_x = {true, false, false};
	// 	ComponentMask side_x_mask(side_x);
	// 	DoFTools::extract_boundary_dofs(dof_handler,
	// 									side_x_mask,
	// 									selected_dofs_x,
	// 									{1});

	// 	// printf current_time, and velocity_qs

	// 	for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
	// 	{
	// 		if (selected_dofs_x[n])
  //     {
	// 			present_solution[n] = (current_time - dT) * velocity_qs;

  //     }
	// 	}

	// 	std::vector<bool> side_yz = {false, true, true};
	// 	ComponentMask side_yz_mask(side_yz);
	// 	DoFTools::extract_boundary_dofs(dof_handler,
	// 									side_yz_mask,
	// 									selected_dofs_yz,
	// 									{1});

	// 	for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
	// 	{
	// 		if (selected_dofs_yz[n])
	// 			present_solution[n] = 0.0;
	// 	}
  // }

  void ElasticProblem::initiate_guess(unsigned int timestep_number)
  {
    // pulling it in y direction 

    std::vector<bool> side_z = {false, false, true};
		ComponentMask side_z_mask(side_z);
    std::vector<bool> selected_dofs_z;

		DoFTools::extract_boundary_dofs(dof_handler,
										side_z_mask,
										selected_dofs_z,
										{1});
    
    // std::vector<bool> side_x = {true, false, false};
		// ComponentMask side_x_mask(side_x);
    // std::vector<bool> selected_dofs_x;

		// DoFTools::extract_boundary_dofs(dof_handler,
		// 								side_x_mask,
		// 								selected_dofs_x,
		// 								{1});

    // double radius_of_cylinder = 7.0e-3;
    double radius_of_cylinder = 100.0e-2/PI;
    double theta = domain_dimensions[0]/radius_of_cylinder;;
    
    double ux_final = radius_of_cylinder * sin(theta);
    double uz_final = radius_of_cylinder * (1 - cos(theta));
		// // printf current_time, and velocity_qs

    if (timestep_number % STEP_OUT == 0 || timestep_number == 1)
      std::cout << ", EndDisp = " << uz_final * (current_time - dT) * velocity_qs << "   " << "--------------------" << std::endl;


		for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
		{
			if (selected_dofs_z[n])
      {
				present_solution[n] = - uz_final * (current_time - dT) * velocity_qs;
      }

      // if (selected_dofs_x[n])
			// {
      //   present_solution[n] = - ux_final * (current_time - dT) * velocity_qs; // this is the displacement of the end point, which should be 0.5mm from the center of the cylinder (so that it's at rest)
      // }
		}


    // maybe give a simple BC of just pulling the end to touch the cylinder
    // bending it over a cylinder of radius 7mm
    
  }




  void ElasticProblem::solve_forward_problem()
  {


    unsigned int timestep_number = 1;
    unsigned int counter = 0;

    current_time = dT;
    
		for (; current_time <= T_final; current_time += dT, ++timestep_number)
		{

      if (timestep_number % STEP_OUT == 0 || timestep_number == 1)
			  std::cout << "time step " << timestep_number << " at t= " << current_time <<", objective = " << objective;

			double last_residual_norm = std::numeric_limits<double>::max();
			counter = 1;

			while ((last_residual_norm > 1.0e-7) && (counter < 10))
			{ // start newton
				
        if (counter == 1)
				{
					initiate_guess(timestep_number);
				}
				// assemble_system_rhs();
        // assemble_system_matrix();
        parallel_assemble_system_rhs();
        parallel_assemble_system_matrix();

				apply_boundaries_and_constraints();
				solve();
				last_residual_norm = compute_residual();

        // if (timestep_number % STEP_OUT == 0)
				//   std::cout << " Iteration : " << counter << "  Residual : " << last_residual_norm << std::endl;

				setup_system(false);
				++counter;
			} // newton iteration done


			//output_forces();
			// calculate_end_disp(current_solution);
			propagate_u();
			// calculate_end_disp(solution_u);

			if (timestep_number % STEP_OUT == 0 || timestep_number == 1)
			{
        // computing fatigue based objective
        // compute_fatigue_data();
        // objective = 0.0;
        // compute_objective();

        std::cout << " Iteration : " << counter << "  Residual : " << last_residual_norm << std::endl;
				output_results((timestep_number));
			}
		}

  }

  void ElasticProblem::propagate_u()
  {
    solution_u =  present_solution;
  }

  double ElasticProblem::compute_residual()
  {
    Vector<double> residual(dof_handler.n_dofs());
		residual = 0.0;

		Vector<double> eval_point(dof_handler.n_dofs());
		eval_point = present_solution;

		FEValues<DIM> fe_values(fe, problemQuadrature,
								update_values | update_gradients |
									update_quadrature_points | update_JxW_values);

		unsigned int n_q_points = problemQuadrature.size();
		const unsigned int dofs_per_cell = fe.dofs_per_cell;

		Vector<double> cell_residual(dofs_per_cell);

    std::vector<std::vector<Tensor<1 , DIM> > > old_solution_gradients(n_q_points, std::vector<Tensor<1,DIM>>(DIM));

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		// const FEValuesExtractors::Vector u(0);

		typename DoFHandler<DIM>::active_cell_iterator cell = dof_handler.begin_active(),
													   endc = dof_handler.end();

		unsigned int cell_index = 0;
    Tensor<2, DIM> PKres; // this is probably piola kirchoff
    Tensor<2, DIM3> grad_u;

		for (; cell != endc; ++cell)
		{

			cell_residual = 0.0;
			cell_index = cell->active_cell_index();

			fe_values.reinit(cell);
			fe_values.get_function_gradients(eval_point, old_solution_gradients); 

			for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{
        
        PKres = 0.0;
        grad_u = 0.0;


        get_gradu_tensor(old_solution_gradients[q_point], grad_u);
        elmMats[cell_index].get_dWdF(grad_u, PKres);				

        for (unsigned int n = 0; n < dofs_per_cell; ++n)
        {
          const unsigned int component_n = fe.system_to_component_index(n).first;

          for(unsigned int j = 0; j < DIM; ++j)
          {
            cell_residual(n) -= PKres[component_n][j]*fe_values.shape_grad(n, q_point)[j]*fe_values.JxW(q_point);
          }
        }

			} // end of q point iteration

			cell->get_dof_indices(local_dof_indices);

			for (unsigned int n = 0; n < dofs_per_cell; ++n)
				residual(local_dof_indices[n]) += cell_residual(n);

    } // end of cell iteration

		constraints.condense(residual);

		std::vector<bool> encastre = {true, true, true};
		ComponentMask encastre_mask(encastre);

		for (types::global_dof_index i :
			 DoFTools::extract_boundary_dofs(dof_handler, encastre_mask, {0, 1}))
			residual(i) = 0.0;

		// At the end of the function, we return the norm of the residual:
		return residual.l2_norm();
  }



  void ElasticProblem::assemble_system_rhs()
  {
    // Assembling the system rhs. I choose to make the rhs and system matrix assemblies separate,
    // because we only do one at a time anyways in the newton method.

    system_rhs = 0.0;

    FEValues<DIM> fe_values (fe, problemQuadrature,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    // unsigned int   n_q_points    = quadrature_formula.size();

    unsigned int n_q_points = problemQuadrature.size();
    double inv_q_points = 1.0/(1.0*n_q_points);

    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<std::vector<Tensor<1 , DIM> > > old_solution_gradients(n_q_points, std::vector<Tensor<1,DIM>>(DIM));

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


    Tensor<2,DIM3> grad_u;
    Tensor<2,DIM3> dW_dF;     // piola kirchoff

    std::vector<Tensor<1, DIM> > rhs_values (n_q_points);

    typename DoFHandler<DIM>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      cell_rhs = 0.0;

      fe_values.reinit (cell);

      fe_values.get_function_gradients(present_solution, old_solution_gradients);

      right_hand_side (fe_values.get_quadrature_points(), rhs_values);  // this is useless since there is no force 

      unsigned int cell_index = cell->active_cell_index();
      von_misses_stress[cell_index] = 0.0;
      // Cauchy1[cell_index] = 0.0;
      // Cauchy2[cell_index] = 0.0;
      // Cauchy3[cell_index] = 0.0;
      // Cauchy12[cell_index] = 0.0;
      // Cauchy13[cell_index] = 0.0;
      // Cauchy23[cell_index] = 0.0;
      

      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
        grad_u = 0.0;
        dW_dF = 0.0;    

        unsigned int indx = cell_index*n_q_points + q_point;
        get_gradu_tensor(old_solution_gradients[q_point], grad_u);
        elmMats[cell_index].get_dWdF(grad_u, dW_dF);

        // calculating the stress field. This is some bullshit method. look for better technqiue
        Tensor<2,DIM3> F_deform_grad, cauchy_stress;
        elmMats[cell_index].get_F(grad_u, F_deform_grad);
        cauchy_stress = contract<1, 0>(dW_dF, transpose(F_deform_grad)) / determinant(F_deform_grad);

        von_misses_stress[cell_index] += compute_von_misses_stress(cauchy_stress)*inv_q_points;

        // assembling cell_matrix


        // assembling system_rhs
        for (unsigned int n = 0; n < dofs_per_cell; ++n)
        {
          const unsigned int component_n = fe.system_to_component_index(n).first;

          for(unsigned int j = 0; j < DIM; ++j)
          {
            cell_rhs(n) -= dW_dF[component_n][j]*fe_values.shape_grad(n, q_point)[j]*fe_values.JxW(q_point);
          }
        }


      }

      // if (cell_index == 1069)
      //   std::cout << "  von misses : " << von_misses_stress[cell_index] <<std::endl;

      cell->get_dof_indices (local_dof_indices);

      for (unsigned int n=0; n<dofs_per_cell; ++n)
        system_rhs(local_dof_indices[n]) -= cell_rhs(n);

    }

    // constraints.condense (system_rhs);

  }


  void ElasticProblem::assemble_system_matrix()
  {
    // Assembling the system matrix. I choose to make the rhs and system matrix assemblies separate,
    // because we only do one at a time anyways in the newton method.

    system_matrix = 0.0;

    FEValues<DIM> fe_values (fe, problemQuadrature,
                          update_values   | update_gradients |
                          update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    // unsigned int   n_q_points    = quadrature_formula.size();

    unsigned int n_q_points = problemQuadrature.size();
    double inv_q_points = 1.0/(1.0*n_q_points);

    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<std::vector<Tensor<1 , DIM> > > old_solution_gradients(n_q_points, std::vector<Tensor<1,DIM>>(DIM));

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

    Tensor<2, DIM3> grad_u;;
    Tensor<4, DIM3> d2W_dF2;     // hessian

    
    typename DoFHandler<DIM>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      cell_matrix = 0.0;

      fe_values.reinit (cell);

      fe_values.get_function_gradients(present_solution, old_solution_gradients);

      unsigned int cell_index = cell->active_cell_index();

      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
        grad_u = 0.0;
        d2W_dF2 = 0.0;    

        unsigned int indx = cell_index*n_q_points + q_point;
        get_gradu_tensor(old_solution_gradients[q_point], grad_u);
        elmMats[cell_index].get_dWdF2(grad_u, d2W_dF2);


        for (unsigned int n = 0; n < dofs_per_cell; ++n)
        {
          const unsigned int component_n = fe.system_to_component_index(n).first;

          for (unsigned int m = 0; m < dofs_per_cell; ++m)
          {
            const unsigned int component_m = fe.system_to_component_index(m).first;

            for (unsigned int j=0; j<DIM; ++j)
              for (unsigned int l=0; l<DIM; ++l)
              {
                cell_matrix(n,m) -= d2W_dF2[component_n][j][component_m][l]*
                        fe_values.shape_grad(n, q_point)[j]*fe_values.shape_grad(m, q_point)[l]*fe_values.JxW(q_point);
              }
          }
        }
      }

      cell->get_dof_indices (local_dof_indices);

      for (unsigned int n=0; n<dofs_per_cell; ++n)
            for (unsigned int m=0; m<dofs_per_cell; ++m)
            {
              system_matrix.add (local_dof_indices[n],
                                 local_dof_indices[m],
                                 cell_matrix(n,m));
            }
    }

    // constraints.condense (system_matrix);

    // std::map<types::global_dof_index,double> boundary_values;

    // std::vector<bool> side2_components = {false, true};
    // ComponentMask side2_mask(side2_components);

    // VectorTools::interpolate_boundary_values (dof_handler,
    //                                           2,
    //                                           ZeroFunction<dim>(dim),
    //                                           boundary_values,
    //                                           side2_mask);

    // MatrixTools::apply_boundary_values (boundary_values,
    //                                     system_matrix,
    //                                     newton_update,
    //                                     system_rhs);
  }

  void ElasticProblem::solve()
  {
    if (dof_handler.n_dofs() < 10000)
		{
			SparseDirectUMFPACK A_direct;
			A_direct.initialize(system_matrix);
			A_direct.vmult(newton_update, system_rhs);
		}
		else
		{
			SolverControl solver_control(dof_handler.n_dofs(), 1e-11);
			SolverCG<> solver(solver_control);

			PreconditionSSOR<SparseMatrix<double>> preconditioner;
			preconditioner.initialize(system_matrix, 1.2);

			solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
		}

		constraints.distribute(newton_update);
		// current_solution.add(steplength,newton_update);
		present_solution += newton_update;
  }



  
void ElasticProblem::apply_boundaries_and_constraints()
  {
    constraints.condense(system_matrix);
		constraints.condense(system_rhs);

		std::map<types::global_dof_index, double> boundary_values;

		std::vector<bool> encastre = {true, true, true};
    std::vector<bool> z_BC = {false, false, true};
		ComponentMask encastre_mask(encastre);
    ComponentMask z_bc_bend(z_BC);

    std::vector<bool> x_BC = {true, false, false};
    ComponentMask x_bc_bend(x_BC);

		VectorTools::interpolate_boundary_values(dof_handler,
												 0,
												 dealii::Functions::ZeroFunction<DIM, double>(DIM),
												 boundary_values,
												 encastre_mask);

		VectorTools::interpolate_boundary_values(dof_handler,
												 1,
												 dealii::Functions::ZeroFunction<DIM, double>(DIM),
												 boundary_values,
												 z_bc_bend);

		// VectorTools::interpolate_boundary_values(dof_handler,
		// 										 1,
		// 										 dealii::Functions::ZeroFunction<DIM, double>(DIM),
		// 										 boundary_values,
		// 										 x_bc_bend);

		MatrixTools::apply_boundary_values(boundary_values,
										   system_matrix,
										   newton_update,
										   system_rhs);
  }

  


  // void ElasticProblem::output_results (const unsigned int cycle) const
  // {

  //   std::vector<std::string> solution_names;
  //   switch (DIM)
  //     {
  //     case 1:
  //       solution_names.push_back ("displacement");
  //       break;
  //     case 2:
  //       solution_names.push_back ("x1_displacement");
  //       solution_names.push_back ("x2_displacement");
  //       break;
  //     case 3:
  //       solution_names.push_back ("x1_displacement");
  //       solution_names.push_back ("x2_displacement");
  //       solution_names.push_back ("x3_displacement");
  //       break;
  //     default:
  //       Assert (false, ExcNotImplemented());
  //       break;
  //     }

  //   std::vector<std::string> solution_names_v;
  //   switch (DIM)
  //     {
  //     case 1:
  //       solution_names_v.push_back ("velocity");
  //       break;
  //     case 2:
  //       solution_names_v.push_back ("x1_velocity");
  //       solution_names_v.push_back ("x2_velocity");
  //       break;
  //     case 3:
  //       solution_names_v.push_back ("x1_velocity");
  //       solution_names_v.push_back ("x2_velocity");
  //       solution_names_v.push_back ("x3_velocity");
  //       break;
  //     default:
  //       Assert (false, ExcNotImplemented());
  //       break;
  //     }

  //   std::vector<std::string> solution_names_a;
  //   switch (DIM)
  //     {
  //     case 1:
  //       solution_names_a.push_back ("accel");
  //       break;
  //     case 2:
  //       solution_names_a.push_back ("x1_accel");
  //       solution_names_a.push_back ("x2_accel");
  //       break;
  //     case 3:
  //       solution_names_a.push_back ("x1_accel");
  //       solution_names_a.push_back ("x2_accel");
  //       solution_names_a.push_back ("x3_accel");
  //       break;
  //     default:
  //       Assert (false, ExcNotImplemented());
  //       break;
  //     }

  //   std::vector<std::string> solutionName_epsp;
  //   solutionName_epsp.push_back("Epsp_eff");
  //   std::vector<std::string> solutionName_ave_p;
  //   solutionName_ave_p.push_back("average_pressure");

  //   // output the total displacements. this requires adding in the uniform solution on top of the displacements

  //   std::string filename0(output_directory);

  //   filename0 += "/lagrangian_solution-";
  //   filename0 += std::to_string(cycle);
  //   filename0 += ".vtu";
  //   std::ofstream output_lagrangian_solution (filename0.c_str());

  //   DataOut<DIM> data_out_lagrangian;

  //   data_out_lagrangian.attach_dof_handler (dof_handler);

  //   std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation (DIM);
  //   interpretation[0] = DataComponentInterpretation::component_is_part_of_vector;
  //   interpretation[1] = DataComponentInterpretation::component_is_part_of_vector;
  //   interpretation[2] = DataComponentInterpretation::component_is_part_of_vector;

  //   data_out_lagrangian.add_data_vector (present_solution, solution_names, DataOut<DIM>::type_dof_data,interpretation);
  //   data_out_lagrangian.add_data_vector (velocity, solution_names_v, DataOut<DIM>::type_dof_data, interpretation);
  //   data_out_lagrangian.add_data_vector (accel, solution_names_a, DataOut<DIM>::type_dof_data, interpretation);
  //   data_out_lagrangian.add_data_vector(dof_handler, present_solution, *postprocess);

  //   data_out_lagrangian.add_data_vector(ave_epsp_eff, solutionName_epsp);
  //   data_out_lagrangian.add_data_vector(ave_pressure, solutionName_ave_p);


  //   data_out_lagrangian.build_patches ();
  //   data_out_lagrangian.write_vtu (output_lagrangian_solution);

  // }


  void ElasticProblem::output_results (const unsigned int cycle) const
  {

    std::vector<std::string> solution_names;
    switch (DIM)
      {
      case 1:
        solution_names.push_back ("displacement");
        break;
      case 2:
        solution_names.push_back ("x1_displacement");
        solution_names.push_back ("x2_displacement");
        break;
      case 3:
        solution_names.push_back ("x1_displacement");
        solution_names.push_back ("x2_displacement");
        solution_names.push_back ("x3_displacement");
        break;
      default:
        Assert (false, ExcNotImplemented());
        break;
      }


    // output the total displacements. this requires adding in the uniform solution on top of the displacements

    std::string filename0(output_directory);

    filename0 += "/lagrangian_solution-";
    filename0 += std::to_string(cycle);
    filename0 += ".vtu";
    std::ofstream output_lagrangian_solution (filename0.c_str());

    std::vector<std::string> solutionName_DIC_ux;
    solutionName_DIC_ux.push_back("phi");
    std::vector<std::string> solutionName_von_misses;
    solutionName_von_misses.push_back("Von_Misses");

    // outputing componenets of stresses
    std::vector<std::string> solutionName_cau1;
    solutionName_cau1.push_back("Cau1");
    std::vector<std::string> solutionName_cau2;
    solutionName_cau2.push_back("Cau2");
    std::vector<std::string> solutionName_cau3;
    solutionName_cau3.push_back("Cau3");
    std::vector<std::string> solutionName_cau12;
    solutionName_cau12.push_back("Cau12");
    std::vector<std::string> solutionName_cau31;
    solutionName_cau31.push_back("Cau31");
    std::vector<std::string> solutionName_cau23;
    solutionName_cau23.push_back("Cau23");

    // fatigue data output
    std::vector<std::string> solutionName_se;
    solutionName_se.push_back("Se");
    std::vector<std::string> solutionName_Nf;
    solutionName_Nf.push_back("Cyc to failure");

    DataOut<DIM> data_out_lagrangian;

    data_out_lagrangian.attach_dof_handler (dof_handler);

    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation (DIM);
    interpretation[0] = DataComponentInterpretation::component_is_part_of_vector;
    interpretation[1] = DataComponentInterpretation::component_is_part_of_vector;
    interpretation[2] = DataComponentInterpretation::component_is_part_of_vector;

    data_out_lagrangian.add_data_vector (solution_u, solution_names, DataOut<DIM>::type_dof_data,interpretation);
    // data_out_lagrangian.add_data_vector (velocity, solution_names_v, DataOut<DIM>::type_dof_data, interpretation);
    // data_out_lagrangian.add_data_vector (accel, solution_names_a, DataOut<DIM>::type_dof_data, interpretation);
    // data_out_lagrangian.add_data_vector(dof_handler, present_solution, *postprocess);

    // data_out_lagrangian.add_data_vector(ave_epsp_eff, solutionName_epsp);
    // data_out_lagrangian.add_data_vector(ave_pressure, solutionName_ave_p);
    data_out_lagrangian.add_data_vector(von_misses_stress, solutionName_von_misses);
    data_out_lagrangian.add_data_vector(phi, solutionName_DIC_ux);
    data_out_lagrangian.add_data_vector(cauchy1, solutionName_cau1);
    data_out_lagrangian.add_data_vector(cauchy2, solutionName_cau2);
    data_out_lagrangian.add_data_vector(cauchy3, solutionName_cau3);
    data_out_lagrangian.add_data_vector(cauchy12, solutionName_cau12);
    data_out_lagrangian.add_data_vector(cauchy12, solutionName_cau31);
    data_out_lagrangian.add_data_vector(cauchy12, solutionName_cau23);

    // adding fatigue data
    data_out_lagrangian.add_data_vector(se_goodman, solutionName_se);
    data_out_lagrangian.add_data_vector(cycles_to_failure, solutionName_Nf);

    data_out_lagrangian.build_patches ();
    data_out_lagrangian.write_vtu (output_lagrangian_solution);

  }




  void ElasticProblem::read_input_file(char* filename)
  {
    FILE* fid;
    int endOfFileFlag;
    char nextLine[MAXLINE];

    int valuesWritten;
    bool fileReadErrorFlag = false;

    grid_dimensions.resize(DIM);
    domain_dimensions.resize(DIM);

    fid = std::fopen(filename, "r");
    if (fid == NULL)
    {
      std::cout << "Unable to open file \"" << filename  << "\"" <<  std::endl;
      fileReadErrorFlag = true;
    }
    else
    {

      // Read in the output name
      char directory_name[MAXLINE];
      getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
      valuesWritten = sscanf(nextLine, "%s", directory_name);
      if (valuesWritten != 1)
      {
        fileReadErrorFlag = true;
        goto fileClose;
      }

      sprintf(output_directory, "output/");
      strcat(output_directory, directory_name);


//      if(objective_type == 0) load_val = 1.0;

      // Read in the grid dimensions
      getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
      valuesWritten = sscanf(nextLine, "%u %u %u", &grid_dimensions[0], &grid_dimensions[1], &grid_dimensions[2]);
      if(valuesWritten != 3)
      {
        fileReadErrorFlag = true;
        goto fileClose;
      }

      // Read in the domain dimensions
      getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
      valuesWritten = sscanf(nextLine, "%lg %lg %lg", &domain_dimensions[0], &domain_dimensions[1],  &domain_dimensions[2]);
      if(valuesWritten != 3)
      {
        fileReadErrorFlag = true;
        goto fileClose;
      }

      // read in the lambda and mu and density
      getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
      valuesWritten = sscanf(nextLine, "%lg %lg %lg", &Shear_mod, &nu_poi_ratio);
      if(valuesWritten != 2)
      {
        fileReadErrorFlag = true;
        goto fileClose;
      }

      // read in the number of guass points in the x and y direction
      getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
      valuesWritten = sscanf(nextLine, "%u  %u %u", &qx, &qy, &qz);
      if(valuesWritten != 3)
      {
        fileReadErrorFlag = true;
        goto fileClose;
      }


      // read in the load value the final time, and the number of steps
      getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
      valuesWritten = sscanf(nextLine, "%lg  %u",  &T_final, &load_steps);
      if(valuesWritten != 2)
      {
        fileReadErrorFlag = true;
        goto fileClose;
      }
      dT = T_final/(1.0*load_steps);
      velocity_qs = 1.0/T_final;

      fileClose:
      {
        fclose(fid);
      }
    }

    if (fileReadErrorFlag)
    {
      // default parameter values
      std::cout << "Error reading input file, Exiting.\n" << std::endl;
      exit(1);
    }
    else
      std::cout << "Input file successfully read" << std::endl;

    // K = ((1.0*DIM)*(lambda) + 2.0*(mu))/(1.0*DIM);

    // make the output directory
    struct stat st;
    if (stat("./output", &st) == -1)
       mkdir("./output", 0700);

    if (stat(output_directory, &st) == -1)
      mkdir(output_directory, 0700);

  }

  void ElasticProblem::getNextDataLine( FILE* const filePtr, char* nextLinePtr,
                          int const maxSize, int* const endOfFileFlag)
  {
    *endOfFileFlag = 0;
    do
    {
      if(fgets(nextLinePtr, maxSize, filePtr) == NULL)
      {
        *endOfFileFlag = 1;
        break;
      }
      while ((nextLinePtr[0] == ' ' || nextLinePtr[0] == '\t') ||
             (nextLinePtr[0] == '\n' || nextLinePtr[0] == '\r' ))
      {
        nextLinePtr = (nextLinePtr + 1);
      }
    }
    while ((strncmp("#", nextLinePtr, 1) == 0) || (strlen(nextLinePtr) == 0));
  }


  ElasticProblem::AssemblyScratchData::
  AssemblyScratchData (const FiniteElement<DIM> &fe, Quadrature<DIM> &quad, const unsigned int step_)
    :
    fe_values (fe,
               quad,
               update_values   | update_gradients |
               update_quadrature_points | update_JxW_values),
    step(step_)
  {}


  ElasticProblem::AssemblyScratchData::
  AssemblyScratchData (const AssemblyScratchData &scratch_data)
    :
    fe_values (scratch_data.fe_values.get_fe(),
               scratch_data.fe_values.get_quadrature(),
               update_values   | update_gradients |
               update_quadrature_points | update_JxW_values),
    step(scratch_data.step)
  {}

  void ElasticProblem::copy_local_to_global_rhs (const RhsAssemblyCopyData &copy_data)
  {
    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
      system_rhs[copy_data.local_dof_indices[i]] += copy_data.cell_rhs[i];

  }


  void ElasticProblem::local_assemble_system_rhs (const typename DoFHandler<DIM>::active_cell_iterator &cell,
      AssemblyScratchData                                  &scratch,
      RhsAssemblyCopyData                                     &copy_data)
  {
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch.fe_values.get_quadrature().size();

    double inv_q_points = 1.0/(1.0*n_q_points);

    Vector<double>       cell_rhs (dofs_per_cell + 1);

    std::vector<std::vector<Tensor<1,DIM> > > old_solution_gradients(n_q_points, std::vector<Tensor<1,DIM>>(DIM));

    cell_rhs = 0.0;

    scratch.fe_values.reinit (cell);

    scratch.fe_values.get_function_gradients(present_solution, old_solution_gradients);
    
    unsigned int cell_index = cell->active_cell_index();
    von_misses_stress[cell_index] = 0.0;
    cauchy1[cell_index] = 0.0;
    cauchy2[cell_index] = 0.0;
    cauchy3[cell_index] = 0.0;
    cauchy12[cell_index] = 0.0;
    cauchy13[cell_index] = 0.0;
    cauchy23[cell_index] = 0.0;

    Tensor<2,DIM3> grad_u;
    Tensor<2,DIM3> dW_dF;     // piola kirchoff
    

    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
    {
      grad_u = 0.0;
      dW_dF = 0.0;    

      unsigned int indx = cell_index*n_q_points + q_point;

      get_gradu_tensor(old_solution_gradients[q_point], grad_u);
      elmMats[cell_index].get_dWdF(grad_u, dW_dF);

      // calculating the stress field. This is some bullshit method. look for better technqiue
      Tensor<2,DIM3> F_deform_grad, cauchy_stress;
      elmMats[cell_index].get_F(grad_u, F_deform_grad);
      cauchy_stress = contract<1, 0>(dW_dF, transpose(F_deform_grad)) / determinant(F_deform_grad);      
      
      von_misses_stress[cell_index] += compute_von_misses_stress(cauchy_stress)*inv_q_points;
      cauchy1[cell_index] += cauchy_stress[0][0]*inv_q_points;
      cauchy2[cell_index] += cauchy_stress[1][1]*inv_q_points;
      cauchy3[cell_index] += cauchy_stress[2][2]*inv_q_points;
      cauchy12[cell_index] += cauchy_stress[0][1]*inv_q_points;
      cauchy13[cell_index] += cauchy_stress[0][2]*inv_q_points;
      cauchy23[cell_index] += cauchy_stress[1][2]*inv_q_points;

      for (unsigned int n = 0; n < dofs_per_cell; ++n)
      {
        const unsigned int component_n = fe.system_to_component_index(n).first;

        for(unsigned int j = 0; j<DIM; ++j)
        {
          cell_rhs(n) -= dW_dF[component_n][j]*scratch.fe_values.shape_grad(n, q_point)[j]*scratch.fe_values.JxW(q_point);
        }
      }
    }


    copy_data.cell_rhs = cell_rhs;

    copy_data.local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices (copy_data.local_dof_indices);

  }

  void ElasticProblem::parallel_assemble_system_rhs()
  {
    system_rhs = 0.0;
    

    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &ElasticProblem::local_assemble_system_rhs,
                    &ElasticProblem::copy_local_to_global_rhs,
                    AssemblyScratchData(fe, problemQuadrature, 0),
                    RhsAssemblyCopyData());

   constraints.condense (system_rhs);

  }




  void ElasticProblem::copy_local_to_global_system_matrix(const MatrixAssemblyCopyData &copy_data)
  {
    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
      for (unsigned int j=0; j<copy_data.local_dof_indices.size(); ++j)
      {
        system_matrix.add(copy_data.local_dof_indices[i],
                          copy_data.local_dof_indices[j],
                          copy_data.cell_matrix[i][j]);
      }
  }
  
  void ElasticProblem::local_assemble_system_matrix(const typename DoFHandler<DIM>::active_cell_iterator &cell,
      AssemblyScratchData                                  &scratch,
      MatrixAssemblyCopyData                                     &copy_data)
  {
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch.fe_values.get_quadrature().size();

    double inv_q_points = 1.0/(1.0*n_q_points);

    std::vector<std::vector<Tensor<1,DIM> > > old_solution_gradients(n_q_points, std::vector<Tensor<1,DIM>>(DIM));

    Tensor<2, DIM3> grad_u;;
    Tensor<4, DIM3> d2W_dF2;     // hessian

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    cell_matrix = 0.0;

    scratch.fe_values.reinit (cell);

    scratch.fe_values.get_function_gradients(present_solution, old_solution_gradients);

    unsigned int cell_index = cell->active_cell_index();

    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
    {
      grad_u = 0.0;
      d2W_dF2 = 0.0;

      unsigned int indx = cell_index*n_q_points + q_point;
      get_gradu_tensor(old_solution_gradients[q_point], grad_u);
      elmMats[cell_index].get_dWdF2(grad_u, d2W_dF2);

      for (unsigned int n = 0; n < dofs_per_cell; ++n)
      {
        const unsigned int component_n = fe.system_to_component_index(n).first;

        for (unsigned int m = 0; m < dofs_per_cell; ++m)
        {
          const unsigned int component_m = fe.system_to_component_index(m).first;

          for (unsigned int j=0; j<DIM; ++j)
            for (unsigned int l=0; l<DIM; ++l)
            {
              cell_matrix(n,m)
                  += (d2W_dF2[component_n][j][component_m][l])*
                                 scratch.fe_values.shape_grad(n, q_point)[j]*scratch.fe_values.shape_grad(m, q_point)[l]
                                 *scratch.fe_values.JxW(q_point);

            }
        }
      }
    }

    copy_data.local_dof_indices.resize(dofs_per_cell);

    cell->get_dof_indices (copy_data.local_dof_indices);
    copy_data.cell_matrix = cell_matrix;

  }


  void ElasticProblem::parallel_assemble_system_matrix()
  {
    system_matrix = 0.0;
    
    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &ElasticProblem::local_assemble_system_matrix,
                    &ElasticProblem::copy_local_to_global_system_matrix,
                    AssemblyScratchData(fe, problemQuadrature, 0),
                    MatrixAssemblyCopyData());
  }




}

#endif // COMPRESSEDSTRIPPACABLOCH_CC_
