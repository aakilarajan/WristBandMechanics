#ifndef CONSTITUITIVE_H_
#define CONSTITUITIVE_H_

#include <iostream>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/tensor.h>
//#include <gsl/gsl_poly.h>

#define DIM3 3
#define DIM 3

using namespace dealii;

  class PowerLaw
  {
    public:

    PowerLaw();
    virtual ~PowerLaw();
    void Update(const double, const double, const double, const double, const double);
    double E(const double &) const;
    double DE(const double &) const;
    double DDE(const double &) const;

    private:

    double f0 = 0.0;
    double C1 = 0.0;
    double r0 = 0.0;
    double x = 0.0;
    double C3 = 0.0;

    double xpp = 0.0;
    double xmm = 0.0;
    double C0 = 0.0;
    double C2 = 0.0;
    double C4 = 0.0;

    PowerLaw(const PowerLaw *);
    PowerLaw * operator = (const PowerLaw *);
  };

  class J2Isotropic
  {
    public:
    J2Isotropic(){};
    void set_modulii(double next_lambda, double next_mu,
        double next_n, double next_m, double next_eps_p_0, double next_eps_p_0_dot, double next_yield);
    virtual ~J2Isotropic(){};
    void set_internal(Tensor<2, DIM3> *next_eps_p, double *next_q, double next_dt)
    {
      dT = next_dt;
      q = next_q;
      eps_p = next_eps_p;
    };
    void update(Tensor<2, DIM3> eps);
    virtual double get_E(Tensor<2,DIM3> &eps);
    virtual void get_dE(Tensor<2,DIM3> &eps, Tensor<2,DIM3> &dW_deps);
    void get_sigma(Tensor<2,DIM3> &eps, Tensor<2,DIM3> &sigma);
    double compute_delta_q(const Tensor<2, DIM3> &Eps, bool output = false) const;
    double get_C(unsigned int i, unsigned int j, unsigned int k, unsigned int l){return C[i][j][k][l];};
    double get_g_E(const double &x){return g.E(x);};
    double get_g_DE(const double &x){return g.DE(x);};

    double get_Wp_E(const double &x){return Wp.E(x);};
    double get_Wp_DE(const double &x){return Wp.DE(x);};
    double get_Wp_DDE(const double &x){return Wp.DDE(x);};

    double get_Wv_E(const double &x){return Wv.E(x);};
    double get_Wv_DE(const double &x){return Wv.DE(x);};
    double get_Wv_DDE(const double &x){return Wv.DDE(x);};

    double get_E_(){return E_;};

    double compute_sigma_M(const Tensor<2,DIM3> &sigma) const;
    void get_deviatoric(const Tensor<2,DIM3> &E, Tensor<2,DIM3> &dev_E) const;

    private:
    double compute_residual(const Tensor<2, DIM3> &Eps, double delta_q) const;
    double compute_slope(const Tensor<2,DIM3> &Eps, double delta_q) const;
    double newton_iterate(const Tensor<2,DIM3> &Eps, const double init_guess,
                          const unsigned int max_steps, const double tol) const;

    PowerLaw Wp;
    PowerLaw Wv;
    PowerLaw g;
    double *q = NULL;
    Tensor<2, DIM3> *eps_p = NULL;
    double dT = 1.0;
    double C[DIM3][DIM3][DIM3][DIM3];
    double lambda = 1.0;
    double mu = 1.0;
    double K = 1.0;
    double eta = 1.0;
//    double k1 = 1.0/3.0;
//    double k2 = 3.0;

    double m = 10.0;
    double n = 5.0;
    double eps_p_0 = 1e-3;
    double eps_p_0_dot = 1.0;
    double yield = 1e-3;

    double E_ = 0.0;

    mutable Tensor<2,DIM3> S_pre;
    mutable Tensor<2,DIM3> M;
    mutable Tensor<2,DIM3> sigma_;
    mutable Tensor<2,DIM3> S_;
    mutable double sigma_M_ = 0.0;
    mutable double sigma_M_pre = 0.0;

  };


  class Compressible_NeoHookean
  {
    public:
    virtual ~Compressible_NeoHookean(){};

    virtual double get_energy(const double nu, const double mu,
                              Tensor<2, DIM3> F, double II_F);
    virtual Tensor<2,DIM3> get_piola_kirchoff_tensor(const double nu, const double mu,
                                            Tensor<2,DIM3> F, Tensor<2,DIM3> F_inv,
                                            double II_F);
    virtual Tensor<4,DIM3> get_incremental_moduli_tensor(const double nu,
                                                const double mu, Tensor<2,DIM3> F_inv,
                                                double II_F);

  };

  class Isothermal_NeoHookean
  {
    public:
    Isothermal_NeoHookean(double thermal_expansion){alpha = thermal_expansion;};

    virtual ~Isothermal_NeoHookean(){};

    virtual double get_energy(const double nu, const double mu,
                              Tensor<2, DIM3> F, double II_F);
    virtual Tensor<2,DIM3> get_piola_kirchoff_tensor(const double nu, const double mu,
                                            Tensor<2,DIM3> F, Tensor<2,DIM3> F_inv,
                                            double II_F);
    virtual Tensor<4,DIM3> get_incremental_moduli_tensor(const double nu,
                                                const double mu, Tensor<2,DIM3> F_inv,
                                                double II_F);

    virtual Tensor<2,DIM3> get_dE_du_dtheta(const double nu, const double mu,
        Tensor<2,DIM3> F, Tensor<2,DIM3> F_inv, double II_F);

    void set_temp(double theta_set)
    {
      theta = theta_set;
      Jth = (1.0 + theta*alpha)*(1.0 + theta*alpha)*(1.0 + theta*alpha);
    };

    void set_alpha(double alpha_set)
    {
      alpha = alpha_set;
    }

    private:

    double theta = 0.0;
    double alpha = 0.0;
    double Jth = 1.0;
  };

  class LinearLagrangian
  {
    public:
    virtual ~LinearLagrangian(){};


    double get_energy(Tensor<4, DIM3> D, Tensor<2, DIM3> E);

    Tensor<2,DIM3> get_piola_kirchoff_tensor(Tensor<4, DIM3> D,
                                            Tensor<2,DIM3> F, Tensor<2,DIM3> E);

    Tensor<4,DIM3> get_incremental_moduli_tensor(Tensor<4, DIM3> D, Tensor<2,DIM3> F,
                                              Tensor<2,DIM3> E);

    Tensor<2, DIM3> get_lagrangian_strain(Tensor<2, DIM3> F);

    Tensor<4,DIM3> get_D(double mu, double nu);

  };

  class IsothermalLinearLagrangian
  {
    public:
      IsothermalLinearLagrangian(double thermal_expansion){alpha = thermal_expansion;};

    virtual ~IsothermalLinearLagrangian(){};

    virtual double get_energy(const double nu, const double mu,
                              Tensor<2, DIM3> &F, Tensor<2, DIM3> &E);
    virtual void get_piola_kirchoff_tensor(const double nu,
                          const double mu, Tensor<2,DIM3> &F, Tensor<2, DIM3> &E, Tensor<2,DIM3> &dW_dF);
    virtual void get_incremental_moduli_tensor(const double nu,
                                             const double mu, Tensor<2,DIM3> &F,
                                                             Tensor<2, DIM3> &E, Tensor<4,DIM3> &d2W_dF);

    virtual void get_dE_du_dtheta(const double nu, const double mu,
                                           Tensor<2,DIM3> &F, Tensor<2, DIM3> &E, Tensor<2,DIM3> &dW_dFdtheta);

    void set_temp(double theta_set)
    {
      theta = theta_set;
      for (unsigned int i = 0; i < DIM3; i ++)
        E_alpha[i][i] = alpha*theta;
    };

    void set_alpha(double alpha_set)
    {
      alpha = alpha_set;
    }

    private:

    double theta = 0.0;
    double alpha = 0.0;
    Tensor<2,DIM3> E_alpha;

  };


  // changing to linear elastic in large deformation case
  
  // class LinearElastic
  // {
  //   public:
  //   LinearElastic(){};

  //   virtual ~LinearElastic(){};
  //   void set_internal(double E, double nu)
  //   {
  //     mu = E/(2.0*(1 + nu));
  //     lambda = E*nu/((1 + nu)*(1 - 2.0*nu));

  //     for (unsigned int i = 0; i < DIM3; i ++)
  //       for (unsigned int j = 0; j < DIM3; j ++)
  //         for (unsigned int k = 0; k < DIM3; k ++)
  //           for (unsigned int l = 0; l < DIM3; l ++)
  //             C[i][j][k][l] = ((i == j) && (k ==l) ? lambda : 0.0) +
  //               ((i == k) && (j ==l) ? mu : 0.0) + ((i == l) && (j ==k) ? mu : 0.0);
  //   };


  //   double get_energy(Tensor<2, DIM3> &grad_u);
  //   Tensor<2, DIM3> get_sigma(Tensor<2, DIM3> &grad_u);
  //   Tensor<4, DIM3> get_C();

  //   Tensor<4, DIM3> C;

  //   private:
  //     double lambda;
  //     double mu;
  // };

  class LinearElastic
  {
    public:

//    virtual ~LinearElastic(){};
	LinearElastic(){};
    LinearElastic(double mu_, double nu_);

    void init(double mu_, double nu_);
    double get_energy(Tensor<2, DIM> &grad_u);
    void get_dWdF(Tensor<2, DIM> &grad_u, Tensor<2, DIM> &dWdF);
    void get_dWdF2(Tensor<2, DIM> &grad_u, Tensor<4, DIM> &dWdF2);

    void get_F(Tensor<2, DIM> &grad_u, Tensor<2, DIM> &F);
    void get_C(Tensor<2, DIM> &grad_u, Tensor<2, DIM> &C);

    double mu;
    double nu;
    double bulk;


    private:
  };





  #endif
