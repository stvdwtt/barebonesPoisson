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
#include <deal.II/lac/parallel_block_vector.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_renumbering.h>

//factor of 5 is 32 elements in 3D, h=0.032
#define refineFactor 5
#define pi (2.0*std::acos(0.0))

#define number_of_components 2

//typedef dealii::parallel::distributed::Vector<double> vectorType;
typedef dealii::LinearAlgebra::distributed::BlockVector<double> vectorType;
typedef dealii::LinearAlgebra::distributed::Vector<double> vectorType_primative;

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
        void vector_value (const dealii::Point<dim> &p,dealii::Vector<double> &vector_IC) const;
        virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
        bool isC;
    };

    // -----------------------------------------------------------------------------------------------------------------------------------------------------------

    template <int dim>
    void InitialCondition<dim>:: vector_value (const dealii::Point<dim> &p,dealii::Vector<double> &vector_IC) const
    {
        vector_IC[0] =  0.0; //-0.999*(std::cos(2.0*pi*p[0])-1.0);

        if (number_of_components > 1)
            vector_IC[1] =  0.0; //-0.999*(std::cos(2.0*pi*p[0])-1.0);

    }

    template <int dim>
    double InitialCondition<dim>::value (const Point<dim> &p, const unsigned int /* component */) const
    {
        //double result = 0.0;
        double result = -0.999*(std::cos(2.0*pi*p[0])-1.0);
        return result;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------------------------------

    template <int dim>
    class InitialConditionC : public Function<dim>
    {
    public:
        InitialConditionC (const unsigned int n_components = 1, bool _isC=true) : Function<dim>(n_components), isC(_isC)
        {
            std::srand(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)+1);
        }
        virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
        bool isC;
    };

    // -----------------------------------------------------------------------------------------------------------------------------------------------------------

    template <int dim>
    double InitialConditionC<dim>::value (const Point<dim> &p, const unsigned int /* component */) const
    {
        double result = std::cos(2.0*pi*p[0]) * std::cos(2.0*pi*p[1]) + 1.0;
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
        number result = -make_vectorized_array(4.0*pi*pi)*std::cos(make_vectorized_array(2.0*pi)*p[0]); // put some trig function in here dealii::MatrixFreeOperators::Based on an MMS solution
        return result;
    }

    /**
   * This class implements the operation of the action of a Laplace matrix,
   * namely $ L_{ij} = \int_\Omega c(\mathbf x) \mathbf \nabla N_i(\mathbf x) \cdot \mathbf \nabla N_j(\mathbf x)\,d \mathbf x$,
   * where $c(\mathbf x)$ is the scalar heterogeneity coefficient.
   *
   * Note that this class only supports the non-blocked vector variant of the
   * dealii::MatrixFreeOperators::Base operator because only a single FEEvaluation object is used in the
   * apply function.
   *
   * @author Denis Davydov, 2016
   */
  template <int dim, int fe_degree, int n_q_points_1d = fe_degree+1, int n_components = 1, typename VectorType = LinearAlgebra::distributed::Vector<double> >
  class CustomLaplaceOperator : public dealii::MatrixFreeOperators::Base<dim, VectorType>
  {
  public:
    /**
     * Number typedef.
     */
    typedef typename dealii::MatrixFreeOperators::Base<dim,VectorType>::value_type value_type;

    /**
     * size_type needed for preconditioner classes.
     */
    typedef typename dealii::MatrixFreeOperators::Base<dim,VectorType>::size_type size_type;

    /**
     * Constructor.
     */
    CustomLaplaceOperator ();

    /**
     * The diagonal is approximated by computing a local diagonal matrix per element
     * and distributing it to the global diagonal. This will lead to wrong results
     * on element with hanging nodes but is still an acceptable approximation
     * to be used in preconditioners.
     */
    virtual void compute_diagonal ();

    /**
     * Set the heterogeneous scalar coefficient @p scalar_coefficient to be used at
     * the quadrature points. The Table should be of correct size, consistent
     * with the total number of quadrature points in <code>dim</code>-dimensions,
     * controlled by the @p n_q_points_1d template parameter. Here,
     * <code>(*scalar_coefficient)(cell,q)</code> corresponds to the value of the
     * coefficient, where <code>cell</code> is an index into a set of cell
     * batches as administered by the MatrixFree framework (which does not work
     * on individual cells, but instead of batches of cells at once), and
     * <code>q</code> is the number of the quadrature point within this batch.
     *
     * Such tables can be initialized by
     * @code
     * std_cxx11::shared_ptr<Table<2, VectorizedArray<double> > > coefficient;
     * coefficient = std_cxx11::make_shared<Table<2, VectorizedArray<double> > >();
     * {
     *   FEEvaluation<dim,fe_degree,n_q_points_1d,1,double> fe_eval(mf_data);
     *   const unsigned int n_cells = mf_data.n_macro_cells();
     *   const unsigned int n_q_points = fe_eval.n_q_points;
     *   coefficient->reinit(n_cells, n_q_points);
     *   for (unsigned int cell=0; cell<n_cells; ++cell)
     *     {
     *       fe_eval.reinit(cell);
     *       for (unsigned int q=0; q<n_q_points; ++q)
     *         (*coefficient)(cell,q) = function.value(fe_eval.quadrature_point(q));
     *     }
     * }
     * @endcode
     * where <code>mf_data</code> is a MatrixFree object and <code>function</code>
     * is a function which provides the following method
     * <code>VectorizedArray<double> value(const Point<dim, VectorizedArray<double> > &p_vec)</code>.
     *
     * If this function is not called, the coefficient is assumed to be unity.
     *
     * The argument to this function is a shared pointer to such a table. The
     * class stores the shared pointer to this table, not a deep copy
     * and uses it to form the Laplace matrix. Consequently, you can update the
     * table and re-use the current object to obtain the action of a Laplace
     * matrix with this updated coefficient. Alternatively, if the table values
     * are only to be filled once, the original shared pointer can also go out
     * of scope in user code and the clear() command or destructor of this class
     * will delete the table.
     */
    void set_coefficient(const std_cxx11::shared_ptr<Table<2, VectorizedArray<value_type> > > &scalar_coefficient );

    virtual void clear();

    /**
     * Read/Write access to coefficients to be used in Laplace operator.
     *
     * The function will throw an error if coefficients are not previously set
     * by set_coefficient() function.
     */
    std_cxx11::shared_ptr< Table<2, VectorizedArray<value_type> > > get_coefficient();

  private:
    /**
     * Applies the laplace matrix operation on an input vector. It is
     * assumed that the passed input and output vector are correctly initialized
     * using initialize_dof_vector().
     */
    virtual void apply_add (VectorType       &dst,
                            const VectorType &src) const;

    /**
     * Applies the Laplace operator on a cell.
     */
    void local_apply_cell (const MatrixFree<dim,value_type>            &data,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int,unsigned int>  &cell_range) const;

    /**
     * Apply diagonal part of the Laplace operator on a cell.
     */
    void local_diagonal_cell (const MatrixFree<dim,value_type>            &data,
                              VectorType                                  &dst,
                              const unsigned int &,
                              const std::pair<unsigned int,unsigned int>  &cell_range) const;

    /**
     * Apply Laplace operator on a cell @p cell.
     */
    void do_operation_on_cell(FEEvaluation<dim,fe_degree,n_q_points_1d,n_components,value_type> &phi,
                              const unsigned int cell) const;

    /**
     * User-provided heterogeneity coefficient.
     */
    std_cxx11::shared_ptr< Table<2, VectorizedArray<value_type> > > scalar_coefficient;
  };

  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
  CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
  CustomLaplaceOperator ()
    :
    dealii::MatrixFreeOperators::Base<dim, VectorType>()
  {
  }



  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
  void
  CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
  clear ()
  {
    dealii::MatrixFreeOperators::Base<dim, VectorType>::clear();
    scalar_coefficient.reset();
  }



  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
  void
  CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
  set_coefficient(const std_cxx11::shared_ptr<Table<2, VectorizedArray<typename dealii::MatrixFreeOperators::Base<dim,VectorType>::value_type> > > &scalar_coefficient_ )
  {
    scalar_coefficient = scalar_coefficient_;
  }



  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
  std_cxx11::shared_ptr< Table<2, VectorizedArray< typename CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::value_type> > >
  CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
  get_coefficient()
  {
    Assert (scalar_coefficient.get(),
            ExcNotInitialized());
    return scalar_coefficient;
  }



  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
  void
  CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
  compute_diagonal()
  {
    typedef typename dealii::MatrixFreeOperators::Base<dim,VectorType>::value_type Number;
    Assert((dealii::MatrixFreeOperators::Base<dim, VectorType>::data.get() != NULL), ExcNotInitialized());

    unsigned int dummy = 0;
    this->inverse_diagonal_entries.
    reset(new DiagonalMatrix<VectorType>());
    VectorType &inverse_diagonal_vector = this->inverse_diagonal_entries->get_vector();
    this->initialize_dof_vector(inverse_diagonal_vector);

    this->data->cell_loop (&CustomLaplaceOperator::local_diagonal_cell,
                           this, inverse_diagonal_vector, dummy);
    this->set_constrained_entries_to_one(inverse_diagonal_vector);

    for (unsigned int i=0; i<inverse_diagonal_vector.local_size(); ++i){
      if (std::abs(inverse_diagonal_vector.local_element(i)) > std::sqrt(std::numeric_limits<Number>::epsilon()))
        inverse_diagonal_vector.local_element(i) = 1./inverse_diagonal_vector.local_element(i);
      else
        inverse_diagonal_vector.local_element(i) = 1.;
    }

    inverse_diagonal_vector.update_ghost_values();
  }



  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
  void
  CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
  apply_add (VectorType       &dst,
             const VectorType &src) const
  {
    dealii::MatrixFreeOperators::Base<dim, VectorType>::data->cell_loop (&CustomLaplaceOperator::local_apply_cell,
                                            this, dst, src);
  }

  namespace
  {
    template<typename Number>
    bool
    non_negative(const VectorizedArray<Number> &n)
    {
      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        if (n[v] < 0.)
          return false;

      return true;
    }
  }



  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
  void
  CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
  do_operation_on_cell(FEEvaluation<dim,fe_degree,n_q_points_1d,n_components,typename dealii::MatrixFreeOperators::Base<dim,VectorType>::value_type> &phi,
                       const unsigned int cell) const
  {
    phi.evaluate (false,true,false);
    if (scalar_coefficient.get())
      {
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          {
            Assert (non_negative((*scalar_coefficient)(cell,q)),
                    ExcMessage("Coefficient must be non-negative"));
            phi.submit_gradient ((*scalar_coefficient)(cell,q)*phi.get_gradient(q), q);
          }
      }
    else
      {
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          {
            phi.submit_gradient (phi.get_gradient(q), q);
          }
      }
    phi.integrate (false,true);
  }




  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
  void
  CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
  local_apply_cell (const MatrixFree<dim,typename dealii::MatrixFreeOperators::Base<dim,VectorType>::value_type> &data,
                    VectorType       &dst,
                    const VectorType &src,
                    const std::pair<unsigned int,unsigned int>  &cell_range) const
  {
    typedef typename dealii::MatrixFreeOperators::Base<dim,VectorType>::value_type Number;
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components,Number> phi (data, this->selected_rows[0]);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        phi.read_dof_values(src);
        do_operation_on_cell(phi,cell);
        phi.distribute_local_to_global (dst);
      }
  }


  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
  void
  CustomLaplaceOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
  local_diagonal_cell (const MatrixFree<dim,typename dealii::MatrixFreeOperators::Base<dim,VectorType>::value_type> &data,
                       VectorType                                       &dst,
                       const unsigned int &,
                       const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    typedef typename dealii::MatrixFreeOperators::Base<dim,VectorType>::value_type Number;
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components,Number> phi (data, this->selected_rows[0]);
    //FEEvaluation<dim,fe_degree,n_q_points_1d,n_components,Number> phi (data);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        VectorizedArray<Number> local_diagonal_vector[phi.tensor_dofs_per_cell*n_components];
        for (unsigned int i=0; i<phi.dofs_per_cell*n_components; ++i)
        //for (unsigned int i=0; i<phi.tensor_dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<phi.dofs_per_cell*n_components; ++j)
            //for (unsigned int j=0; j<phi.tensor_dofs_per_cell; ++j)
              phi.begin_dof_values()[j] = VectorizedArray<Number>();
            phi.begin_dof_values()[i] = 1.;
            do_operation_on_cell(phi,cell);
            local_diagonal_vector[i] = phi.begin_dof_values()[i];
          }
          for (unsigned int i=0; i<phi.tensor_dofs_per_cell*n_components; ++i){
              phi.begin_dof_values()[i] = local_diagonal_vector[i];
          }
        phi.distribute_local_to_global (dst);
      }
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
        typedef CustomLaplaceOperator<dim,degree_finite_element,degree_finite_element+1,number_of_components> SystemMatrixType;
        // dealii::MatrixFreeOperators::LaplaceOperator<dim,degree_finite_element,degree_finite_element+1,2> SystemMatrixType;

        void setup_system ();
        void solve ();
        void output_results (const unsigned int cycle) const;


        void local_apply_rhs (const MatrixFree<dim,double> &data, vectorType &dst,
            const vectorType &src,
            const std::pair<unsigned int,unsigned int> &cell_range) const;

        void calc_invM (vectorType_primative & invM) const;

        SourceTerm<dim> source_term;

        dealii::parallel::distributed::Triangulation<dim> triangulation;
        FE_Q<dim>                        fe_phi_scalar;
        FESystem<dim>                    fe_phi;
        FE_Q<dim>                        fe_c;
        FESystem<dim>                    fe; //  (FE_Q<dim>(QGaussLobatto<1>(2)),1);
        DoFHandler<dim>                  dof_handler;
        DoFHandler<dim>                  dof_handler_phi;
        DoFHandler<dim>                  dof_handler_c;
        std::vector<const DoFHandler<dim>*> dof_vector;
        ConstraintMatrix                 constraints_phi;
        ConstraintMatrix                 constraints_c;
        ConstraintMatrix                 constraints;
        std::vector<const ConstraintMatrix *> constraints_vector;
        IndexSet                         locally_relevant_dofs, locally_relevant_dofs_phi, locally_relevant_dofs_c;
        SystemMatrixType                 system_matrix;
        vectorType                       Source_Term, Phi, C;
        vectorType_primative            invM;
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
    fe_phi_scalar(1),
    fe_phi(fe_phi_scalar,2),
    fe_c(1),
    fe (fe_phi_scalar,2,fe_c,1), // Question: Why doesn't this have to match the quadrature rule I put into the matrix free object? This is like Step 37 in this regard
    dof_handler (triangulation),
    dof_handler_phi (triangulation),
    dof_handler_c (triangulation),
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
        dof_handler_phi.distribute_dofs (fe_phi);
        dof_handler_c.distribute_dofs (fe_c);
        //

        std::vector<unsigned int> sub_blocks(number_of_components + 1, 0);         // initialize vector to zeros, dim for vel plus one more for p; labels sets of components in the block
        sub_blocks[number_of_components] = 1;
        dealii::DoFRenumbering::component_wise(dof_handler, sub_blocks);

        std::vector<types::global_dof_index> dofs_per_block(2);
        DoFTools::count_dofs_per_block(dof_handler, dofs_per_block, sub_blocks); // Get the number of dofs in each block, not quite sure how this works

        pcout << "Number of global active cells: " << triangulation.n_global_active_cells() << std::endl;
        pcout << "Number of degrees of freedom:  " << dof_handler.n_dofs() << std::endl;
        constraints_phi.clear();
        constraints_c.clear();
        constraints.clear();

        DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);
        DoFTools::extract_locally_relevant_dofs (dof_handler_phi, locally_relevant_dofs_phi);
        DoFTools::extract_locally_relevant_dofs (dof_handler_c, locally_relevant_dofs_c);
        constraints.reinit (locally_relevant_dofs);
        constraints_phi.reinit (locally_relevant_dofs_phi);
        constraints_c.reinit (locally_relevant_dofs_c);

        DoFTools::make_hanging_node_constraints (dof_handler, constraints);
        DoFTools::make_hanging_node_constraints (dof_handler_phi, constraints_phi);
        DoFTools::make_hanging_node_constraints (dof_handler_c, constraints_c);

        //FIXME: Check nonhomogenous Dirichlet BC
        std::vector<bool> component_mask;
        //component_mask.push_back(true);
        //component_mask.push_back(true);
        VectorTools::interpolate_boundary_values (dof_handler_phi, 0, ZeroFunction<dim>(number_of_components), constraints_phi,component_mask);
        VectorTools::interpolate_boundary_values (dof_handler_phi, 1, ZeroFunction<dim>(number_of_components), constraints_phi,component_mask);

        constraints.close();
        constraints_phi.close();
        constraints_c.close();

        setup_time += time.wall_time();
        pcout << "Distribute DoFs & B.C.      "
        <<  time.wall_time() << "s" << std::endl;
        time.restart();

        //data structures
        {
        dof_vector.push_back(&dof_handler_phi);
        dof_vector.push_back(&dof_handler_c);
        constraints_vector.push_back(&constraints_phi);
        constraints_vector.push_back(&constraints_c);

        typename MatrixFree<dim,double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim,double>::AdditionalData::partition_partition;

        additional_data.mapping_update_flags = ( update_values | update_gradients | update_JxW_values |
            update_quadrature_points);

        std_cxx11::shared_ptr<MatrixFree<dim,double> > system_mf_storage(new MatrixFree<dim,double>());

        system_mf_storage->reinit (dof_vector, constraints_vector, QGauss<1>(fe.degree+1),additional_data);

        system_matrix.initialize (system_mf_storage);
        }

        double memory=system_matrix.memory_consumption();
        memory=Utilities::MPI::sum(memory, MPI_COMM_WORLD);
        pcout  << "System matrix memory consumption:     "
        << memory*1e-6
        << " MB."
        << std::endl;

        // Now the solution and RHS vectors

        Phi.reinit(2);
        Source_Term.reinit(2);
        for (unsigned int d = 0; d < 2; ++d)
        {
            Phi.block(d).reinit(dofs_per_block[d]);
            Source_Term.block(d).reinit(dofs_per_block[d]);             // Setting the size of the vector in each block
        }
        Phi.collect_sizes();
        Source_Term.reinit(Phi);

        invM.reinit(Phi.block(1));

        memory=Phi.memory_consumption();
        memory=Utilities::MPI::sum(memory, MPI_COMM_WORLD);

        pcout  << "Vector memory consumption:     "
        << 8*memory*1e-6
        << " MB."
        << std::endl;

        //Initial Condition
        VectorTools::interpolate (dof_handler_phi,InitialCondition<dim> (number_of_components),Phi.block(0));
        VectorTools::interpolate (dof_handler_c,InitialConditionC<dim> (),Phi.block(1));

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
            FEEvaluation<dim,degree_finite_element,degree_finite_element+1, number_of_components> mat(data,0);
            //loop over all "cells"
            for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
            {
                mat.reinit (cell);
                mat.read_dof_values(src.block(0));
                mat.evaluate (true,false,false);
                for (unsigned int q=0; q<mat.n_q_points; ++q){
                    Tensor<1, number_of_components, VectorizedArray<double> > source_term_tensor;
                    source_term_tensor[0] = source_term.value(mat.quadrature_point(q));
                    source_term_tensor[1] = source_term.value(mat.quadrature_point(q));
                    mat.submit_value(source_term_tensor,q);
                }
                mat.integrate (true,false);
                mat.distribute_local_to_global (dst.block(0));

            }

            FEEvaluation<dim,degree_finite_element,degree_finite_element+1, 1> mat2(data,1);
            //loop over all "cells"
            for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
            {
                mat2.reinit (cell);
                mat2.read_dof_values(src.block(1));
                mat2.evaluate (true,true,false);
                for (unsigned int q=0; q<mat2.n_q_points; ++q){
                    mat2.submit_value(mat2.get_value(q)+make_vectorized_array(0.0),q);
                    mat2.submit_gradient(-make_vectorized_array(0.001)*mat2.get_gradient(q),q);
                }
                mat2.integrate (true,true);
                mat2.distribute_local_to_global (dst.block(1));
            }
        }

    template <int dim>
    void PoissonProblem<dim>::
    calc_invM (vectorType_primative & invM) const
    {
        dealii::VectorizedArray<double> one = dealii::make_vectorized_array (1.0);
        dealii::FEEvaluation<dim,1> fe_eval(*(system_matrix.get_matrix_free()), 1);
        const unsigned int n_q_points = fe_eval.n_q_points;
        for (unsigned int cell=0; cell<system_matrix.get_matrix_free()->n_macro_cells(); ++cell){
            fe_eval.reinit(cell);
            for (unsigned int q=0; q<n_q_points; ++q){
                fe_eval.submit_value(one,q);
            }
            fe_eval.integrate (true,false);
            fe_eval.distribute_local_to_global (invM);
        }

        invM.compress(dealii::VectorOperation::add);
        //invert mass matrix diagonal elements
        for (unsigned int k=0; k<invM.local_size(); ++k){
            if (std::abs(invM.local_element(k))>1.0e-15){
                invM.local_element(k) = 1./invM.local_element(k);
            }
            else{
                invM.local_element(k) = 0;
            }
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

        // Take explicit time step (ignoring invM for now)
        calc_invM(invM);
        Source_Term.block(1).scale(invM);
        Phi.block(1) = Source_Term.block(1);

        std::vector<unsigned int> selected_row_blocks;
        selected_row_blocks.push_back(0);
        typename MatrixFree<dim,double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim,double>::AdditionalData::partition_partition;

        additional_data.mapping_update_flags = ( update_values | update_gradients | update_JxW_values |
            update_quadrature_points);

        std_cxx11::shared_ptr<MatrixFree<dim,double> > system_mf_storage(new MatrixFree<dim,double>());

        system_mf_storage->reinit (dof_vector, constraints_vector, QGauss<1>(fe.degree+1),additional_data);
        system_matrix.initialize (system_mf_storage,selected_row_blocks);

        // Now calculate the LHS (actually done in vmult)
        IterationNumberControl           solver_control (1000, 1.0e-6,true);
        SolverCG<vectorType_primative>              solver (solver_control);


        // Set up the Jacobi preconditioner
        system_matrix.compute_diagonal();
        PreconditionJacobi<SystemMatrixType> precondition;
        precondition.initialize(system_matrix,1.0);

        //solver.solve(system_matrix,Phi.block(0),Source_Term.block(0),PreconditionIdentity());
        solver.solve(system_matrix,Phi.block(0),Source_Term.block(0),precondition);

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
        //data_out.attach_dof_handler (dof_handler);
        std::vector<DataComponentInterpretation::DataComponentInterpretation> partofvector;
        if (number_of_components == 1){
            partofvector.push_back(DataComponentInterpretation::component_is_scalar);
        }
        else {
            partofvector.push_back(DataComponentInterpretation::component_is_part_of_vector);
            partofvector.push_back(DataComponentInterpretation::component_is_part_of_vector);
        }
        std::vector<std::string> output_name;
        output_name.push_back("phi");
        if (number_of_components > 1)
            output_name.push_back("phi");
        data_out.add_data_vector (dof_handler_phi, Phi.block(0), output_name,partofvector);

        partofvector.clear();
        output_name.clear();
        partofvector.push_back(DataComponentInterpretation::component_is_scalar);
        output_name.push_back("c");
        data_out.add_data_vector (dof_handler_c, Phi.block(1), output_name,partofvector);

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
