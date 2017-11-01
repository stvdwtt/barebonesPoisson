// Matrix Free implementation of the Poisson equation
#include <deal.II/base/quadrature.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
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
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/operators.h>

//factor of 5 is 32 elements in 3D, h=0.032
#define refineFactor 7
#define pi (2.0*std::acos(0.0))

//typedef dealii::parallel::distributed::Vector<double> vectorType;
typedef dealii::LinearAlgebra::distributed::Vector<double> vectorType;

namespace matFreePoisson
{
    using namespace dealii;
    const unsigned int degree_finite_element = 1;
    const unsigned int dimension = 2;

    // ===========================================================================================================================================================
    // Inital condition
    // ===========================================================================================================================================================
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

    // -----------------------------------------------------------------------------------------------------------------------------------------------------------

    template <int dim>
    double InitialCondition<dim>::value (const Point<dim> &p, const unsigned int /* component */) const
    {
        //double result = 0.0;
        double result = -0.999*(std::cos(2.0*pi*p[0])-1.0);
        return result;
    }

    // ===========================================================================================================================================================
    // Source term (i.e. the RHS)
    // ===========================================================================================================================================================
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
    // -----------------------------------------------------------------------------------------------------------------------------------------------------------

    template <int dim>
    template <typename number>
    number SourceTerm<dim>::value (const Point<dim,number> &p, const unsigned int /* component */) const
    {
        number result = -make_vectorized_array(4.0*pi*pi)*std::cos(make_vectorized_array(2.0*pi)*p[0]); // put some trig function in here based on an MMS solution
        return result;
    }

    // ===========================================================================================================================================================
    // Now the actual Poisson Class
    // ===========================================================================================================================================================
    template <int dim>
    class PoissonProblem
    {
    public:
        PoissonProblem ();
        void run ();
    private:
        typedef dealii::MatrixFreeOperators::LaplaceOperator<dim,degree_finite_element> SystemMatrixType;

        void setup_system ();
        void solve ();
        void output_results (const unsigned int cycle) const;


        void local_apply_rhs (const MatrixFree<dim,double> &data, vectorType &dst,
            const vectorType &src,
            const std::pair<unsigned int,unsigned int> &cell_range) const;

        SourceTerm<dim> source_term;

        dealii::parallel::distributed::Triangulation<dim> triangulation;
        FE_Q<dim>                        fe;
        DoFHandler<dim>                  dof_handler;
        ConstraintMatrix                 constraints;
        IndexSet                         locally_relevant_dofs;
        SystemMatrixType                 system_matrix;
        vectorType                       Source_Term, Phi;
        double                           setup_time;
        ConditionalOStream               pcout;
    };

    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
    // constructor
    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
    template <int dim>
    PoissonProblem<dim>::PoissonProblem ()
    :
    triangulation (MPI_COMM_WORLD),
    fe (QGaussLobatto<1>(degree_finite_element+1)), // Question: Why doesn't this have to match the quadrature rule I put into the matrix free object? This is like Step 37 in this regard
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

        //FIXME: Check nonhomogenous Dirichlet BC
        VectorTools::interpolate_boundary_values (dof_handler, 0, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values (dof_handler, 1, ZeroFunction<dim>(), constraints);
        constraints.close();
        setup_time += time.wall_time();
        pcout << "Distribute DoFs & B.C.      "
        <<  time.wall_time() << "s" << std::endl;
        time.restart();

        //data structures
        {
        typename MatrixFree<dim,double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim,double>::AdditionalData::partition_partition;

        additional_data.mapping_update_flags = ( update_values | update_gradients | update_JxW_values |
            update_quadrature_points);

        std_cxx11::shared_ptr<MatrixFree<dim,double> > system_mf_storage(new MatrixFree<dim,double>());

        system_mf_storage->reinit (dof_handler, constraints, QGauss<1>(fe.degree+1),additional_data);

        system_matrix.initialize (system_mf_storage);
        }

        double memory=system_matrix.memory_consumption();
        memory=Utilities::MPI::sum(memory, MPI_COMM_WORLD);
        pcout  << "System matrix memory consumption:     "
        << memory*1e-6
        << " MB."
        << std::endl;

        system_matrix.initialize_dof_vector(Source_Term);
        Phi.reinit  (Source_Term);
        memory=Phi.memory_consumption();
        memory=Utilities::MPI::sum(memory, MPI_COMM_WORLD);

        pcout  << "Vector memory consumption:     "
        << 8*memory*1e-6
        << " MB."
        << std::endl;

        //Initial Condition
        VectorTools::interpolate (dof_handler,InitialCondition<dim> (1),Phi);

        //timing
        setup_time += time.wall_time();
        pcout << "Setup matrix-free system:    "
        << time.wall_time() << "s" << std::endl;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
    //Implement finite element operator application
    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
    template <int dim>
    void PoissonProblem<dim>::
    local_apply_rhs (const MatrixFree<dim,double>               &data,
        vectorType                                 &dst,
        const vectorType                           &src,
        const std::pair<unsigned int,unsigned int> &cell_range) const
        {
            FEEvaluation<dim,degree_finite_element> mat(data);
            //loop over all "cells"
            for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
            {
                mat.reinit (cell);
                mat.read_dof_values(src);
                mat.evaluate (true,false,false);
                for (unsigned int q=0; q<mat.n_q_points; ++q){
                    mat.submit_value(source_term.value(mat.quadrature_point(q)),q);
                }
                mat.integrate (true,false);
                mat.distribute_local_to_global (dst);
            }
        }


    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
    //solve
    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
    template <int dim>
    void PoissonProblem<dim>::solve () {

        Timer time;
        char buffer[200];

        // First calculate the RHS
        Source_Term = 0.0;
        system_matrix.get_matrix_free()->cell_loop(&PoissonProblem::local_apply_rhs,this,Source_Term,Phi);

        // Now calculate the LHS (actually done in vmult)
        IterationNumberControl           solver_control (1000, 1.0e-6,true);
        SolverCG<vectorType>              solver (solver_control);


        // Set up the Jacobi preconditioner
        system_matrix.compute_diagonal();
        PreconditionJacobi<SystemMatrixType> precondition;
        precondition.initialize(system_matrix,1.0);

        //solver.solve(system_matrix,Phi,Source_Term,PreconditionIdentity());
        solver.solve(system_matrix,Phi,Source_Term,precondition);

        pcout << "time in CGSolve: " << time.wall_time() << "s" <<std::endl;

        sprintf(buffer, "initial residual:%12.6e, current residual:%12.6e, nsteps:%u, tolerance criterion:%12.6e, solution: %12.6e\n", \
        Source_Term.l2_norm(),			\
        solver_control.last_value(),				\
        solver_control.last_step(), solver_control.tolerance(), Phi.l2_norm());
        pcout<<buffer;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
    //output
    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
    template <int dim>
    void PoissonProblem<dim>::output_results (const unsigned int cycle) const {

        //constraints.distribute (C0);
        Phi.update_ghost_values();
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (Phi, "phi");
        data_out.build_patches ();

        const std::string filename = "solution-" + Utilities::int_to_string (cycle, 6);
        std::ofstream output ((filename + "." + Utilities::int_to_string (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),4) + ".vtu").c_str());
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

    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
    //run
    // -----------------------------------------------------------------------------------------------------------------------------------------------------------
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

// ===========================================================================================================================================================
// main
// ===========================================================================================================================================================
int main (int argc, char **argv)
{
    using namespace matFreePoisson;
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,dealii::numbers::invalid_unsigned_int);
    try
    {
        dealii::deallog.depth_console(4);
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
