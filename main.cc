// Matrix Free implementation of the Poisson equation
#include <deal.II/base/quadrature.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/distributed/tria.h>
#include <fstream>
#include <sstream>


//factor of 5 is 32 elements in 3D, h=0.032
#define refineFactor 8
#define pi (2.0*std::acos(0.0))

typedef dealii::parallel::distributed::Vector<double> vectorType;

namespace matFreePoisson
{
  using namespace dealii;
  const unsigned int degree_finite_element = 1;
  const unsigned int dimension = 2;

  // Source term (i.e. the RHS)
  template <int dim>
  class SourceTerm : public Function<dim>
  {
  public:
    SourceTerm (const unsigned int n_components = 1, bool _isC=true) : Function<dim>(n_components), isC(_isC)
    {
      std::srand(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)+1);
    }
    template<typename number>
    number value (const Point<dim,number> &p, const unsigned int component = 0) const;
    bool isC;
  };
  template <int dim>
  template <typename number>
  number SourceTerm<dim>::value (const Point<dim,number> &p, const unsigned int /* component */) const
  {
    number result = -make_vectorized_array(4.0*pi*pi)*std::cos(make_vectorized_array(2.0*pi)*p[0]); // put some trig function in here based on an MMS solution
    return result;
  }

  //Matrix-free implementation
  template <int dim, int fe_degree, typename number>
  class PoissonOperator : public Subscriptor
  {
  public:
    PoissonOperator ();
    void clear();
    void reinit (const DoFHandler<dim>  &dof_handler, const ConstraintMatrix  &constraints, const unsigned int level = numbers::invalid_unsigned_int);
    unsigned int m () const;
    unsigned int n () const;
    void invMK (vectorType &dst, const vectorType &src) const;
    void getK (vectorType &dst, const vectorType &src) const;
    void getRHS  (vectorType &dst, const vectorType &src) const;
    void vmult (vectorType &dst, const vectorType &src) const;
    number el (const unsigned int row, const unsigned int col) const;
    void set_diagonal (const vectorType &diagonal);
    std::size_t memory_consumption () const;
    vectorType *X;
    vectorType invM;

    SourceTerm<dim> source_term;

  private:
    void local_apply (const MatrixFree<dim,number> &data, vectorType &dst,
		      const vectorType &src,
		      const std::pair<unsigned int,unsigned int> &cell_range) const;
    void local_apply_rhs (const MatrixFree<dim,number> &data, vectorType &dst,
          	const vectorType &src,
            const std::pair<unsigned int,unsigned int> &cell_range) const;
    MatrixFree<dim,number>      data;
  };

  //Constructor
  template <int dim, int fe_degree, typename number>
  PoissonOperator<dim,fe_degree,number>::PoissonOperator ():Subscriptor()
  {}

  //Matrix free data structure size
  template <int dim, int fe_degree, typename number>
  unsigned int PoissonOperator<dim,fe_degree,number>::m () const
  {
    return data.get_vector_partitioner()->size();
  }
  template <int dim, int fe_degree, typename number>
  unsigned int PoissonOperator<dim,fe_degree,number>::n () const
  {
    return data.get_vector_partitioner()->size();
  }
  //Reset Matrix free data structure
  template <int dim, int fe_degree, typename number>
  void PoissonOperator<dim,fe_degree,number>::clear ()
  {
    data.clear();
  }
  //datastructure memory consumption estimation
  template <int dim, int fe_degree, typename number>
  std::size_t PoissonOperator<dim,fe_degree,number>::memory_consumption () const
  {
    return (data.memory_consumption ());
   }

  //Initialize Matrix Free data structure
  template <int dim, int fe_degree, typename number>
  void PoissonOperator<dim,fe_degree,number>::reinit (const DoFHandler<dim>  &dof_handler, const ConstraintMatrix  &constraint, const unsigned int level)
  {
    typename MatrixFree<dim,number>::AdditionalData additional_data;
    additional_data.mpi_communicator = MPI_COMM_WORLD;
    additional_data.tasks_parallel_scheme = MatrixFree<dim,number>::AdditionalData::partition_partition;
    additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
    QGaussLobatto<1> quadrature (fe_degree+1);
    data.reinit (dof_handler, constraint, quadrature, additional_data);

    //Compute  invM
    data.initialize_dof_vector (invM);
    VectorizedArray<double> one = make_vectorized_array (1.0);
    //Select gauss lobatto quad points which are suboptimal but give diogonal M
    FEEvaluation<dim,fe_degree> fe_eval(data);
    const unsigned int            n_q_points = fe_eval.n_q_points;
    for (unsigned int cell=0; cell<data.n_macro_cells(); ++cell)
      {
	fe_eval.reinit(cell);
	for (unsigned int q=0; q<n_q_points; ++q)
	  fe_eval.submit_value(one,q);
	fe_eval.integrate (true,false);
	fe_eval.distribute_local_to_global (invM);
      }
    invM.compress(VectorOperation::add);
    //
    for (unsigned int k=0; k<invM.local_size(); ++k)
      if (std::abs(invM.local_element(k))>1e-15)
	invM.local_element(k) = 1./invM.local_element(k);
      else
	invM.local_element(k) = 0;
  }

  //Implement finite element operator application
  template <int dim, int fe_degree, typename number>
  void PoissonOperator<dim,fe_degree,number>::
  local_apply (const MatrixFree<dim,number>               &data,
               vectorType                                 &dst,
               const vectorType                           &src,
               const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,fe_degree> mat(data);
    //loop over all "cells"
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        mat.reinit (cell);
        mat.read_dof_values(src);
	mat.evaluate (false,true,false);
	for (unsigned int q=0; q<mat.n_q_points; ++q){
	  mat.submit_gradient(-mat.get_gradient(q),q); // minus sign added for Poisson
	}
	mat.integrate (false,true);
	mat.distribute_local_to_global (dst);
      }
  }

  //Implement finite element operator application
  template <int dim, int fe_degree, typename number>
  void PoissonOperator<dim,fe_degree,number>::
  local_apply_rhs (const MatrixFree<dim,number>               &data,
      vectorType                                 &dst,
      const vectorType                           &src,
      const std::pair<unsigned int,unsigned int> &cell_range) const
      {
          FEEvaluation<dim,fe_degree> mat(data);
          //loop over all "cells"
          for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
          {
              mat.reinit (cell);
              mat.read_dof_values(src);
              mat.evaluate (true,false,false);
              for (unsigned int q=0; q<mat.n_q_points; ++q){
                  mat.submit_value(source_term.value(mat.quadrature_point(q)),q);
                  //mat.submit_value(-4.0*pi*pi*std::cos(2.0*pi*mat.quadrature_point(q)[0]),q);
                  //mat.submit_value(make_vectorized_array(1.0),q);
              }
              mat.integrate (true,false);
              mat.distribute_local_to_global (dst);
          }
      }

  // Matrix free data structure vmult operations. (STILL NEEDS TO BE MODIFIED!)
  template <int dim, int fe_degree, typename number>
  void PoissonOperator<dim,fe_degree,number>::vmult (vectorType &dst, const vectorType &src) const
  {
    // Poisson
    getK(dst,src);

    // Cahn-Hilliard
    // invMK(*X,src); //X=M^(-1)*K*src
    // invMK(dst,*X); //dst=M^(-1)*K*X
    // dst*=(mobility*dt*lambda); //dst*=(M*dt*Lambda)
    // dst+=src;
  }
  template <int dim, int fe_degree, typename number>
  void PoissonOperator<dim,fe_degree,number>::invMK (vectorType &dst, const vectorType &src) const
  {
    dst=0.0;
    //perform w=K*v
    data.cell_loop (&PoissonOperator::local_apply, this, dst, src);
    //perform q=M^-1*w...to give q=M^-1*K*v
    for (unsigned int k=0; k<invM.local_size(); ++k)
      dst.local_element(k)*=invM.local_element(k);
  }

  template <int dim, int fe_degree, typename number>
  void PoissonOperator<dim,fe_degree,number>::getK (vectorType &dst, const vectorType &src) const
  {
    dst=0.0;
    //perform w=K*v
    data.cell_loop (&PoissonOperator::local_apply, this, dst, src);
  }

  template <int dim, int fe_degree, typename number>
  void PoissonOperator<dim,fe_degree,number>::getRHS (vectorType &dst, const vectorType &src) const
  {
    dst=0.0;
    //perform w=K*v
    data.cell_loop (&PoissonOperator::local_apply_rhs, this, dst, src);
  }




  //Inital condition
  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    InitialCondition (const unsigned int n_components = 1, bool _isC=true) : Function<dim>(n_components), isC(_isC)
    {
      std::srand(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)+1);
    }
    virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
    bool isC;
  };
  template <int dim>
  double InitialCondition<dim>::value (const Point<dim> &p, const unsigned int /* component */) const
  {
    double result = 0.0;
    //double result = 1.0 * std::cos(2.0*pi*p[0]);
    return result;
  }

  //
  //Now the actual Cahn Hilliard Class
  //
  template <int dim>
  class PoissonProblem
  {
  public:
    PoissonProblem ();
    void run ();
  private:
    typedef PoissonOperator<dim,degree_finite_element,double> SystemMatrixType;
    void setup_system ();
    void solve ();
    void output_results (const unsigned int cycle) const;
    dealii::parallel::distributed::Triangulation<dim> triangulation;
    FE_Q<dim>                        fe;
    DoFHandler<dim>                  dof_handler;
    ConstraintMatrix                 constraints;
    IndexSet                         locally_relevant_dofs;
    SystemMatrixType                 system_matrix;
    vectorType                       C0, Mu, X, X1, R, P, H, Source_Term, Phi;
    double                           setup_time;
    unsigned int                     increment;
    ConditionalOStream               pcout;
  };

  //constructor
  template <int dim>
  PoissonProblem<dim>::PoissonProblem ()
    :
    triangulation (MPI_COMM_WORLD),
    fe (QGaussLobatto<1>(degree_finite_element+1)),
    dof_handler (triangulation),
    pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
  {}

  //setup
  template <int dim>
  void PoissonProblem<dim>::setup_system ()
  {
    Timer time;
    time.start ();
    setup_time = 0;
    //
    system_matrix.clear();
    dof_handler.distribute_dofs (fe);
    //
    pcout << "Number of global active cells: " << triangulation.n_global_active_cells() << std::endl;
    pcout << "Number of degrees of freedom:  " << dof_handler.n_dofs() << std::endl;
    DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);
    constraints.clear();
    constraints.reinit (locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints (dof_handler, constraints);

    //FIXME: Check Dirichlet BC
    VectorTools::interpolate_boundary_values (dof_handler, 0, ZeroFunction<dim>(), constraints);
    VectorTools::interpolate_boundary_values (dof_handler, 1, ZeroFunction<dim>(), constraints);
    constraints.close();
    setup_time += time.wall_time();
    pcout << "Distribute DoFs & B.C.      "
	  <<  time.wall_time() << "s" << std::endl;
    time.restart();
    //data structures
    system_matrix.reinit (dof_handler, constraints);
    system_matrix.X=&X1;
    double memory=system_matrix.memory_consumption();
    memory=Utilities::MPI::sum(memory, MPI_COMM_WORLD);
    pcout  << "System matrix memory consumption:     "
	   << memory*1e-6
	   << " MB."
	   << std::endl;
    C0.reinit (system_matrix.invM);
    Mu.reinit (C0);
    X.reinit  (C0);
    X1.reinit (C0);
    R.reinit  (C0);
    P.reinit  (C0);
    H.reinit  (C0);
    Source_Term.reinit  (C0);
    Phi.reinit  (C0);
    memory=C0.memory_consumption();
    memory=Utilities::MPI::sum(memory, MPI_COMM_WORLD);
    pcout  << "Vector memory consumption:     "
	   << 8*memory*1e-6
	   << " MB."
	   << std::endl;
    //Initial Condition
    VectorTools::interpolate (dof_handler,InitialCondition<dim> (1),Phi);
    Mu=0.0; X=0.0;
    //timing
    setup_time += time.wall_time();
    pcout << "Setup matrix-free system:    "
	   << time.wall_time() << "s" << std::endl;
  }

  //solve
  template <int dim>
  void PoissonProblem<dim>::solve ()
  {
    Timer time;
    char buffer[200];

    // Begin Poisson solve
    Timer time1;

    // First calculate the RHS
    system_matrix.getRHS(Source_Term,Phi);

    //VectorTools::interpolate (dof_handler,SourceTerm<dim> (1),Source_Term);

    // Now calculate the LHS (actually done in vmult)
    IterationNumberControl           solver_control (1000, 1.0e-14);
    SolverCG<vectorType>              solver (solver_control);
    solver.solve(system_matrix,Phi,Source_Term,PreconditionIdentity());

    pcout << "time in CGSolve: " << time.wall_time() << "s" <<std::endl;

    sprintf(buffer, "initial residual:%12.6e, current residual:%12.6e, nsteps:%u, tolerance criterion:%12.6e, solution: %12.6e\n", \
                Source_Term.l2_norm(),			\
                solver_control.last_value(),				\
                solver_control.last_step(), solver_control.tolerance(), Phi.l2_norm());
                pcout<<buffer;

    //Phi = Source_Term;
    // End Poisson solve


    //Begin solve
    //compute f(c0)+lambda*M^(-1)*K*c0
    // Timer time1;
    // system_matrix.invMK(X,C0); //M^(-1)*K*c0
    // double c0;
    // for (unsigned int k=0; k<C0.local_size(); ++k){
    //   c0=C0.local_element(k);
    //   X.local_element(k)=fcV+ lambda*X.local_element(k); //f(c0)+lambda*M^(-1)*K*c0
    // }
    // pcout << "time in RHS: " << time1.wall_time() << "s"<<std::endl;
    // //(1 + mobility*dt*lambda*M^(-1)*K*M^(-1)*K)Mu=f(c0)+lambda*M^(-1)*K*c0
    // //cg.solve(system_matrix,Mu,X,PreconditionIdentity());
    // SolverControl           solver_control (1000, 1e-13*X.l2_norm());
    // SolverCG<vectorType>              solver (solver_control);
    // solver.solve(system_matrix,Mu,X,PreconditionIdentity());
    //
    // //c=c0-mobility*dt*M^(-1)*K*mu
    // system_matrix.invMK(X,Mu);
    // X*=(mobility*dt);
    // C0-=X;
    // pcout << "time in CGSolve: " << time.wall_time() << "s" <<std::endl;
  }

  //output
  template <int dim>
  void PoissonProblem<dim>::output_results (const unsigned int cycle) const
  {
    //constraints.distribute (C0);
    Phi.update_ghost_values();
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (Phi, "phi");
    data_out.build_patches ();

    const std::string filename = "solution-" + Utilities::int_to_string (cycle, 6);
    std::ofstream output ((filename +
                         "." + Utilities::int_to_string (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),4) + ".vtu").c_str());
    data_out.write_vtu (output);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
	std::vector<std::string> filenames;
	for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes (MPI_COMM_WORLD); ++i)
	  filenames.push_back ("solution-" +
			       Utilities::int_to_string (cycle, 6) + "." +
			       Utilities::int_to_string (i, 4) + ".vtu");
      std::ofstream master_output ((filename + ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
      }
    pcout << "Output written to:" << filename.c_str() << "\n\n";
  }

  //run
  template <int dim>
  void PoissonProblem<dim>::run ()
  {
    Timer time;

    GridGenerator::hyper_cube (triangulation, 0., 1.);
    triangulation.refine_global (refineFactor);

    // Mark the boundaries
    typename Triangulation<dim>::cell_iterator
	cell = triangulation.begin (),
	endc = triangulation.end();

	for (; cell!=endc; ++cell){

		// Mark all of the faces
		for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell;++face_number){
			for (unsigned int i=0; i<dim; i++){
				if ( std::fabs(cell->face(face_number)->center()(i) - (0)) < 1e-12 ){
					cell->face(face_number)->set_boundary_id (2*i);
				}
				else if (std::fabs(cell->face(face_number)->center()(i) - (1.0)) < 1e-12){
					cell->face(face_number)->set_boundary_id (2*i+1);
				}

			}
		}
	}

    setup_system();
    output_results(0);
    solve ();
	output_results (1);

    pcout << "Total  time    "  << time.wall_time() << "s" << std::endl;
  }

}

//main
int main (int argc, char **argv)
{
  using namespace matFreePoisson;
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,dealii::numbers::invalid_unsigned_int);
  try
    {
      dealii::deallog.depth_console(0);
      PoissonProblem<dimension> poisson_problem;
      poisson_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
