#include "NonLinearOpt.h"
#include <fstream>

using namespace dealii;
int main ()
{
  compressed_strip::ElasticProblem ep;

  char fileName[MAXLINE];
  std::cout << "Please enter an input file: " << std::endl;
  std::cin >> fileName;
  ep.read_input_file(fileName);

  ep.load_state(0);

  ep.setup_system();

  std::cout << "\n   Number of active cells:       "
            << ep.get_number_active_cells()
            << std::endl;


  std::cout << "   Number of degrees of freedom: "
            << ep.get_n_dofs()
            << std::endl << std::endl;

  // output inital mesh
  ep.output_results (0);
  std::string filename(ep.output_directory);
  filename += "/objective_data.out";
  FILE* objective_out;
  objective_out = std::fopen(filename.c_str(), "w");
  fclose(objective_out);
  ep.save_current_state(1, true);



  return(0);
}
