#ifndef CONSTITUITIVE_CC_
#define CONSTITUITIVE_CC_

#include "Constituitive.h"

#define DIM3 3

using namespace dealii;


  // J2 Isotropic
  void J2Isotropic::set_modulii(double next_lambda, double next_mu,
      double next_n, double next_m, double next_eps_p_0, double next_eps_p_0_dot, double next_yield)
  {

    m = next_m;
    n = next_n;
    eps_p_0 = next_eps_p_0;
    eps_p_0_dot = next_eps_p_0_dot;
    yield = next_yield;

    lambda = next_lambda;
    mu = next_mu;

    for (unsigned int i = 0; i < DIM3; i ++)
       for (unsigned int j = 0; j < DIM3; j ++)
         for (unsigned int k = 0; k < DIM3; k ++)
           for (unsigned int l = 0; l < DIM3; l ++)
             C[i][j][k][l] = ((i == j) && (k ==l) ? lambda : 0.0) +
               ((i == k) && (j ==l) ? mu : 0.0) + ((i == l) && (j ==k) ? mu : 0.0);

    K = ((1.0*DIM3)*(lambda) + 2.0*(mu))/(1.0*DIM3);

    double Wpf0 = yield * mu;
    double WpC1 =  yield * mu;
    double Wpr0 = eps_p_0;
    double Wpx = 1.0/n;
    double WpC3 = 0.0;

    double Wvf0 = 0.0;
    double WvC1 = yield* mu;
    double Wvr0 = eps_p_0_dot;
    double Wvx = 1.0/m;
    double WvC3 = 0.0;

    double gr0 = WvC1;
    double gC1 = Wvr0;
    double gx = 1.0/Wvx;
    double gf0 = 0.0;
    double gC3 = 0.0;

    Wp.Update(Wpf0, WpC1, Wpr0, Wpx, WpC3);
    Wv.Update(Wvf0, WvC1, Wvr0, Wvx, WvC3);
    g.Update(gf0, gC1, gr0, gx, gC3);
  }

  void J2Isotropic::update(Tensor<2, DIM3> Eps)
  {
    double dq = compute_delta_q(Eps);

    if(dq > 0.0)
    {
      *q += dq;

      get_deviatoric((Eps - *eps_p), S_pre);
      S_pre *= 2.0*(mu);

      double sigma_M_pre = compute_sigma_M(S_pre);

      M = (1.5/sigma_M_pre)*S_pre;
      *eps_p +=  dq*M;
    }

  }

  double J2Isotropic::get_E(Tensor<2, DIM3> &Eps)
  {
    double W = 0.0;

    double dq = compute_delta_q(Eps);

    double W_p = Wp.E(*q + dq);

//    Tensor<2,DIM3> S_pre;
    get_deviatoric((Eps - *eps_p), S_pre);
    S_pre *= 2.0*(mu);
    double sigma_M_pre = compute_sigma_M(S_pre);

//    Tensor<2,DIM3> M = (1.5/sigma_M_pre)*S_pre;
    M = (1.5/sigma_M_pre)*S_pre;
    Tensor<2,DIM3> Epsp = *eps_p + dq*M;

    Tensor<2,DIM3> Eps_e = Eps - Epsp;
    double W_e = 0.0;
    for (unsigned int l = 0; l<DIM3; l++)
      for (unsigned int k = 0; k<DIM3; k++)
        for (unsigned int j = 0; j<DIM3; j++)
          for (unsigned int i = 0; i<DIM3; i++)
            W_e += 0.5*C[i][j][k][l]*Eps_e[i][j]*Eps_e[k][l];


    W = W_e + W_p + dT*Wv.E(dq/dT);

    return W;
  }

  void J2Isotropic::get_dE(Tensor<2, DIM3> &Eps, Tensor<2,DIM3> &dW_deps)
  {
    dW_deps = 0.0;

//    Tensor<2,DIM3> S_pre;
    get_deviatoric((Eps - *eps_p), S_pre);
    S_pre *= 2.0*(mu);

    sigma_M_pre = compute_sigma_M(S_pre);

//    Tensor<2,DIM3> M = (1.5/sigma_M_pre)*S_pre;
    M = (1.5/sigma_M_pre)*S_pre;

    double dq = compute_delta_q(Eps);
    double P = K*trace(Eps);

    dW_deps = S_pre - 2.0*mu*dq*M;
    for (unsigned int j = 0; j < DIM3; j++)
        dW_deps[j][j] += P;

    if (dq > 0.0)
    {
      *q += dq;
      *eps_p += dq*M;
    }
  }

  void J2Isotropic::get_sigma(Tensor<2, DIM3> &Eps, Tensor<2,DIM3> &sigma)
  {
    sigma = 0.0;
    for(unsigned int i = 0; i < DIM3; i ++)
      for(unsigned int j = 0; j < DIM3; j ++)
        for(unsigned int k = 0; k < DIM3; k ++)
          for(unsigned int l = 0; l < DIM3; l ++)
            sigma[i][j] += C[i][j][k][l]*(Eps[k][l] - (*eps_p)[k][l]);

  }

  double J2Isotropic::compute_delta_q(const Tensor<2,DIM3> &Eps, bool output) const
  {

    double lambda_1 = -compute_residual(Eps, 0.0);
    double delta_q;
    if(output)
      std::cout << "  LAMBDA1 :" << lambda_1<< std::endl;

    if(lambda_1 > 0.0)
      delta_q = 0.0;
    else
      delta_q = newton_iterate(Eps, 1e-10, 1000, 1e-7);

    return delta_q;
  }

  double J2Isotropic::newton_iterate(const Tensor<2,DIM3> &Eps, const double init_guess,
                           const unsigned int max_steps, const double tol) const
  {
    unsigned int iteration = 0;
    double current_dq = init_guess;
    double current_residual = compute_residual(Eps, current_dq);

    // Loops until coverge or go over max iterations
    while(fabs(current_residual/lambda) > tol &&
             iteration < max_steps)
    {
      // get slope
      double dR = compute_slope(Eps, current_dq);

      // solve for the newton step
      double d_dq = (-current_residual/dR);

      // add the step
      double alpha = 1.0;
      while(current_dq + alpha*d_dq < 0.0)
        alpha = alpha*0.5;

      current_dq += alpha*d_dq;

      current_residual = compute_residual(Eps, fabs(current_dq));
      iteration ++;

    }

    if(iteration >= max_steps)
    {
      std::cout << "My Newton : Max Iterations Reached. Exiting."<< std::endl;
      exit(-1);
    }

    return current_dq;
  }

  double J2Isotropic::compute_residual(const Tensor<2, DIM3> &Eps, double delta_q) const
  {
    double P = K*trace(Eps);
//    Tensor<2, DIM3> S_pre;
//    get_deviatoric(Eps - *eps_p, S_pre);
//    S_pre *= 2.0*(mu);
//
//    double sigma_M_pre = compute_sigma_M(S_pre);
//
////    Tensor<2, DIM3> M = (1.5/sigma_M_pre)*S_pre;
//    M = (1.5/sigma_M_pre)*S_pre;

//    Tensor<2, DIM3> sigma = S_pre - 2.0*(mu)*delta_q*M;
    sigma_ = S_pre - 2.0*(mu)*delta_q*M;
    for (unsigned int i = 0; i < DIM3; i ++)
      for (unsigned int j = 0; j < DIM3; j ++)
        sigma_[i][j] += (i == j ? P : 0.0);

    sigma_M_ = compute_sigma_M(sigma_);
    if(sigma_M_ == -1.0)
      sigma_M_ = 0.0;

    double R = sigma_M_ - Wp.DE(*q + delta_q) - Wv.DE(delta_q/dT);

    return R;
  }

  double J2Isotropic::compute_slope(const Tensor<2,DIM3> &Eps, double delta_q) const
  {
    double P = K*trace(Eps);
//    Tensor<2, DIM3> S_pre;
//    get_deviatoric(Eps - *eps_p, S_pre);
//    S_pre *= 2.0*(mu);
//
//    double sigma_M_pre = compute_sigma_M(S_pre);
//
////    Tensor<2, DIM3> M = (1.5/sigma_M_pre)*S_pre;
//    M = (1.5/sigma_M_pre)*S_pre;

//    Tensor<2, DIM3> sigma = S_pre - 2.0*(mu)*delta_q*M;
//    sigma_ = S_pre - 2.0*(mu)*delta_q*M;
//    for (unsigned int i = 0; i < DIM3; i ++)
//      for (unsigned int j = 0; j < DIM3; j ++)
//        sigma_[i][j] += (i == j ? P : 0.0);
//
//    double sigma_M = compute_sigma_M(sigma_);

//    Tensor<2,DIM3> S;
    get_deviatoric(sigma_, S_);
//    Tensor<2, DIM3> dSigma_dq = -2.0*(mu)*M;
//    Tensor<2, DIM3> dS_dq;
//    get_deviatoric(dSigma_dq, dS_dq);

    double dSigma_M_dq = 0.0;
    for (unsigned int i = 0; i < DIM3; i ++)
      for (unsigned int j = 0; j < DIM3; j ++)
        dSigma_M_dq += S_[i][j]*-2.0*(mu)*M[i][j];

    dSigma_M_dq *= 1.5/sigma_M_;

    double dR = dSigma_M_dq - Wp.DDE(*q + delta_q) - Wv.DDE(delta_q/dT)/dT;

    return dR;
  }

  double J2Isotropic::compute_sigma_M(const Tensor<2,DIM3> &sigma) const
  {
    Tensor<2,DIM3> S;
    get_deviatoric(sigma, S);
    double sigma_M = 0.0;
    for (unsigned int i = 0; i < DIM3; i ++)
      for (unsigned int j = 0; j < DIM3; j ++)
        sigma_M += S[i][j]*S[i][j];

    sigma_M = sqrt(1.5*sigma_M);
    if(sigma_M < 1e-14)
      sigma_M = -1.0;

    return sigma_M;
  }

  void J2Isotropic::get_deviatoric(const Tensor<2,DIM3> &E, Tensor<2,DIM3> &Dev_E) const
  {
    Dev_E = E;
    double dev_factor  = 1.0/(1.0*DIM3);
    double Tr_E = trace(E);
    for (unsigned int i = 0; i < DIM3; i ++)
      for (unsigned int j = 0; j < DIM3; j ++)
        Dev_E[i][j] -= (i == j ? dev_factor*Tr_E : 0.0);
  }





  // Compressible Neo hookean

  inline
  double Compressible_NeoHookean::get_energy(const double nu, const double mu,
                              Tensor<2, DIM3> F, double II_F)
  {
    double I_C = F[1][0]*F[1][0] + F[0][0]*F[0][0] + F[0][1]*F[0][1] + F[1][1]*F[1][1];
    double W = mu*(0.5*(I_C - 2 - log(II_F*II_F)) + (nu/(1.0- nu))*(II_F - 1.0)*(II_F - 1.0));

    return W;
  }


  inline
  Tensor<2,DIM3> Compressible_NeoHookean::get_piola_kirchoff_tensor(const double nu, const double mu,
                                            Tensor<2,DIM3> F, Tensor<2,DIM3> F_inv,
                                            double II_F)
  {
    Tensor<2, DIM3> tmp;

    for (unsigned int i=0; i<DIM3; ++i)
      for (unsigned int j=0; j<DIM3; ++j)
      {
        tmp[i][j] = mu*F[i][j] - mu*F_inv[j][i] +
                    (2.0*mu*nu/(1.0- nu))*(II_F*II_F - II_F)*F_inv[j][i];
      }

    return tmp;
  }

  inline
  Tensor<4,DIM3> Compressible_NeoHookean::get_incremental_moduli_tensor(const double nu,
                                                const double mu, Tensor<2,DIM3> F_inv,
                                                double II_F)
  {

    Tensor<4,DIM3> tmp;

    for (unsigned int i=0; i<DIM3; ++i)
      for (unsigned int j=0; j<DIM3; ++j)
        for (unsigned int k=0; k<DIM3; ++k)
          for (unsigned int l=0; l<DIM3; ++l)
          {
            tmp[i][j][k][l] = ((i==k) && (j==l) ? mu : 0.0) +
                (mu - (2.0*mu*nu/(1.0-nu))*(II_F*II_F - II_F))*F_inv[j][k]*F_inv[l][i] +
                (4.0*nu*mu/(1.0-nu))*(II_F*II_F - 0.5*II_F)*F_inv[l][k]*F_inv[j][i];
          }

    return tmp;
  }


  // Isothermal NeoHookean
  inline
  double Isothermal_NeoHookean::get_energy(const double nu, const double mu,
                              Tensor<2, DIM3> F, double II_F)
  {
    double I_C = F[1][0]*F[1][0] + F[0][0]*F[0][0] + F[0][1]*F[0][1] + F[1][1]*F[1][1];
    double W = mu*(0.5*(I_C - 2 - log(II_F*II_F/(Jth*Jth))) + (nu/(1.0- nu))*(II_F/Jth - 1.0)*(II_F/Jth - 1.0));

    return W;
  }


  inline
  Tensor<2,DIM3> Isothermal_NeoHookean::get_piola_kirchoff_tensor(const double nu, const double mu,
                                            Tensor<2,DIM3> F, Tensor<2,DIM3> F_inv,
                                            double II_F)
  {
    Tensor<2, DIM3> tmp;

    double Jth_inv = 1.0/Jth;
    double Jth_inv_sq = Jth_inv*Jth_inv;
    for (unsigned int i=0; i<DIM3; ++i)
      for (unsigned int j=0; j<DIM3; ++j)
      {
        tmp[i][j] = mu*F[i][j] - mu*F_inv[j][i] +
                    (2.0*mu*nu/(1.0- nu))*(II_F*II_F*Jth_inv_sq - II_F*Jth_inv)*F_inv[j][i];
      }

    return tmp;

  }

  inline
  Tensor<4,DIM3> Isothermal_NeoHookean::get_incremental_moduli_tensor(const double nu,
                                                const double mu, Tensor<2,DIM3> F_inv,
                                                double II_F)
  {

    Tensor<4,DIM3> tmp;
    double Jth_inv = 1.0/Jth;
    double Jth_inv_sq = Jth_inv*Jth_inv;
    for (unsigned int i=0; i<DIM3; ++i)
      for (unsigned int j=0; j<DIM3; ++j)
        for (unsigned int k=0; k<DIM3; ++k)
          for (unsigned int l=0; l<DIM3; ++l)
          {
            tmp[i][j][k][l] = ((i==k) && (j==l) ? mu : 0.0) +
                (mu - (2.0*mu*nu/(1.0-nu))*(II_F*II_F*Jth_inv_sq - II_F*Jth_inv))*F_inv[j][k]*F_inv[l][i] +
                (4.0*nu*mu/(1.0-nu))*(II_F*II_F*Jth_inv_sq - 0.5*II_F*Jth_inv)*F_inv[l][k]*F_inv[j][i];
          }

    return tmp;
  }

  inline
  Tensor<2, DIM3> Isothermal_NeoHookean::get_dE_du_dtheta(const double nu,
            const double mu, Tensor<2,DIM3> F, Tensor<2,DIM3> F_inv, double II_F)
  {
    Tensor<2, DIM3> tmp;
    double Jth_inv = 1.0/Jth;
    double Jth_inv_sq = Jth_inv*Jth_inv;
    for (unsigned int i=0; i<DIM3; ++i)
      for (unsigned int j=0; j<DIM3; ++j)
      {
        tmp[i][j] = mu*F[i][j] - mu*F_inv[j][i] +
                    (2.0*mu*nu/(1.0- nu))*(-2.0*II_F*II_F*Jth_inv_sq*Jth_inv + II_F*Jth_inv*Jth_inv)*
                    3.0*alpha*(1.0 + alpha*theta)*(1.0 + alpha*theta)*F_inv[j][i];
      }

    return tmp;
  }


  // The linear one

  inline
  double LinearLagrangian::get_energy(Tensor<4, DIM3> D,
                              Tensor<2, DIM3> E)
  {
    double W = 0.0;


    for (unsigned int i=0; i<DIM3; ++i)
      for (unsigned int j=0; j<DIM3; ++j)
        for (unsigned int k=0; k<DIM3; ++k)
          for (unsigned int l=0; l<DIM3; ++l)
          {
            W += 0.5*D[i][j][k][l]*E[k][l]*E[i][j];
          }


    return W;
  }

  inline
  Tensor<2,DIM3> LinearLagrangian::get_piola_kirchoff_tensor(Tensor<4, DIM3> D,
                                            Tensor<2,DIM3> F, Tensor<2,DIM3> E)
  {

    Tensor<2, DIM3> tmp;

    for (unsigned int i=0; i<DIM3; ++i)
      for (unsigned int j=0; j<DIM3; ++j)
        for (unsigned int n=0; n<DIM3; ++n)
          for (unsigned int l=0; l<DIM3; ++l)
            for (unsigned int m=0; m<DIM3; ++m)
            {
              tmp[i][j] += F[i][j]*D[j][l][m][n]*E[m][n];
            }



    return tmp;
  }

  inline
  Tensor<4,DIM3> LinearLagrangian::get_incremental_moduli_tensor(Tensor<4, DIM3> D, Tensor<2,DIM3> F, Tensor<2,DIM3> E)
  {
    Tensor<4, DIM3> tmp;

    for (unsigned int i=0; i<DIM3; ++i)
      for (unsigned int j=0; j<DIM3; ++j)
        for (unsigned int p=0; p<DIM3; ++p)
          for (unsigned int q=0; q<DIM3; ++q)
            for (unsigned int m=0; m<DIM3; ++m)
              for (unsigned int n=0; n<DIM3; ++n)
              {
                tmp[i][j][p][q] += (i == p ? 1.0:0.0)*D[j][q][m][n]*E[m][n] +
                                     F[i][m]*D[j][m][n][p]*F[n][q] + F[i][m]*D[j][m][p][n]*F[n][q];
              }




    return tmp;
  }

  inline
  Tensor<2, DIM3> LinearLagrangian::get_lagrangian_strain(Tensor<2, DIM3> F)
  {
    Tensor<2,DIM3> E;
    for (unsigned int i=0; i<DIM3; ++i)
    {
      for (unsigned int j=0; j<DIM3; ++j)
      {
        for (unsigned int m=0; m<DIM3; ++m)
        {
          E[i][j] += F[i][m]*F[j][m];
        }
      }
      E[i][i] += -1.0;
    }

    return E;
  }

  inline
  Tensor<4,DIM3> LinearLagrangian::get_D(double mu, double nu)
  {
    Tensor<4, DIM3> tmp;

    double scalingFactor = mu/(1.0 - nu*nu);
    double val = (1 - nu)/4.0;

    tmp[1][1][1][1] = 1.0;
    tmp[1][1][2][2] = nu;
    tmp[2][2][1][1] = nu;
    tmp[2][2][2][2] = 1.0;
    tmp[1][2][1][2] = val;
    tmp[1][2][2][1] = val;
    tmp[2][1][1][2] = val;
    tmp[2][1][2][1] = val;

    tmp *= scalingFactor;

    return tmp;
  }

  inline
  double IsothermalLinearLagrangian::get_energy(const double nu, const double mu,
                                          Tensor<2, DIM3> &F, Tensor<2, DIM3> &E)
  {
    double W = 0.0;

    Tensor<2,DIM3> Et = E - E_alpha;
    double trEt = trace(Et);
    double tr_Et_sq = Et.norm_square();
    double lambda = (2*mu*nu)/(1.0 - 2.0*nu);

    W = (lambda/2.0)*trEt*trEt + mu*tr_Et_sq;

    return W;
  }

  inline
  void IsothermalLinearLagrangian::get_piola_kirchoff_tensor(const double nu,
      const double mu, Tensor<2,DIM3> &F, Tensor<2, DIM3> &E, Tensor<2,DIM3> &dW_dF)
  {
    dW_dF = 0.0;
    Tensor<2,DIM3> Et = E - E_alpha;
    double trEt = trace(Et);
    double lambda = (2*mu*nu)/(1.0 - 2.0*nu);

    for(unsigned int i = 0; i < DIM3; i ++)
      for(unsigned int j = 0; j < DIM3; j ++)
      {
        dW_dF[i][j] += lambda*trEt*F[i][j];
        for(unsigned int k = 0; k < DIM3; k ++)
          dW_dF[i][j] += 2.0*mu*Et[k][j]*F[i][k];
      }
  }

  inline
  void IsothermalLinearLagrangian::get_incremental_moduli_tensor(const double nu,
      const double mu, Tensor<2,DIM3> &F, Tensor<2, DIM3> &E, Tensor<4, DIM3> &d2W_dFdF)
  {
    d2W_dFdF = 0.0;
    Tensor<2,DIM3> Et = E - E_alpha;
    double trEt = trace(Et);
    double lambda = (2*mu*nu)/(1.0 - 2.0*nu);

    for(unsigned int i = 0; i < DIM3; i ++)
      for(unsigned int j = 0; j < DIM3; j ++)
        for(unsigned int k = 0; k < DIM3; k ++)
          for(unsigned int l = 0; l < DIM3; l ++)
          {
            d2W_dFdF[i][j][k][l] += lambda*F[i][j]*F[k][l] + mu*F[i][l]*F[k][j] +
                              (i == k && j == l ? lambda*trEt : 0.0) + 2.0*mu*Et[l][j]*(i == k ? 1.0 : 0.0);
            d2W_dFdF[i][j][k][j] += mu*F[i][l]*F[k][l];
          }
  }

  inline
  void IsothermalLinearLagrangian::get_dE_du_dtheta(const double nu,
      const double mu, Tensor<2,DIM3> &F, Tensor<2, DIM3> &E, Tensor<2,DIM3> &dW_dFdTheta)
  {
    double lambda = (2*mu*nu)/(1.0 - 2.0*nu);

    dW_dFdTheta = -(3.0*lambda + 2.0*mu)*alpha*F;
  }


  // Linear Elasticity but in large deformation
  
  // inline
  // double LinearElastic::get_energy(Tensor<2, DIM3> &grad_u)
  // {
  //   double W = 0.0;

  //   Tensor<2, DIM3> epsilon;
  //   for (unsigned int i = 0; i < DIM3; i ++)
  //     for (unsigned int j = 0; j < DIM3; j ++)
  //     {
  //       epsilon[i][j] = 0.5*(grad_u[i][j] + grad_u[j][i]);
  //       W += epsilon[i][j]*epsilon[i][j];
  //     }

  //   W *= mu;
  //   W += 0.5*(lambda)*trace(epsilon)*trace(epsilon);

  //   return W;
  // }

  // Tensor<2, DIM3> LinearElastic::get_sigma(Tensor<2, DIM3> &grad_u)
  // {
  //   Tensor<2, DIM3> sig;
  //   Tensor<2, DIM3> epsilon;
  //   Tensor<2, DIM3> Id;
  //   for (unsigned int i = 0; i < DIM3; i ++)
  //     for (unsigned int j = 0; j < DIM3; j ++)
  //     {
  //       epsilon[i][j] = 0.5*(grad_u[i][j] + grad_u[j][i]);

  //       if(i == j) Id[i][j] = 1.0;
  //     }

  //   sig = lambda*trace(epsilon)*Id + 2.0*(mu)*epsilon;


  //   return sig;
  // }

  // Tensor<4, DIM3> LinearElastic::get_C()
  // {
  //   return C;
  // }


	LinearElastic::LinearElastic(double mu_, double nu_)
	{
		  mu = mu_;
		  nu = nu_;
		  bulk = 2*mu*(1+nu)/(3*(1-2*nu));
	}

	void LinearElastic::init(double mu_, double nu_)
	{
		  mu = mu_;
		  nu = nu_;
		  bulk = 2*mu*(1+nu)/(3*(1-2*nu));
	}



  double LinearElastic::get_energy(Tensor<2, DIM> &grad_u)
  {
    double W = 0.0;
    Tensor<2,DIM> F;

    for (unsigned int t = 0; t < DIM; t++){
        for (unsigned int v = 0; v < DIM; v ++){
         F[t][v] = (t==v) ? grad_u[t][v]+1.0 : grad_u[t][v];
        }}

    Tensor<2, DIM> C = contract<1,0>( transpose(F), F );
    W = 0.5*mu*( trace(C)-3-2*log(determinant(F)) ) + 0.5*bulk*( determinant(F)-1 )*( determinant(F)-1 ); // double_contract<0,0,1,1>(double_contract<2, 0, 3, 1>(C, eps), eps);

    return W;
  }

  void LinearElastic::get_dWdF(Tensor<2, DIM> &grad_u, Tensor<2, DIM> &dWdF)
  {

	  Tensor<2,DIM> F;

		  for (unsigned int t = 0; t < DIM; t++){
			  for (unsigned int v = 0; v < DIM; v ++){
				  F[t][v] = (t==v) ? grad_u[t][v]+1.0 : grad_u[t][v];
			  }}

	  dWdF = mu*( F - transpose(invert(F)) ) + bulk*determinant(F)*(determinant(F)-1)*transpose(invert(F));

  }

  void LinearElastic::get_dWdF2(Tensor<2, DIM> &grad_u, Tensor<4, DIM> &dWdF2)
  {

	  Tensor<2,DIM> F, Finv;

	  for (unsigned int t = 0; t < DIM; t++){
		  for (unsigned int v = 0; v < DIM; v ++){
			  F[t][v] = (t==v) ? grad_u[t][v]+1.0 : grad_u[t][v];
		  }}

	  Finv = invert(F); double J = determinant(F);

	  for (unsigned int i = 0; i < DIM; i++){
	 		  for (unsigned int j = 0; j < DIM; j++){
		 		  for (unsigned int k = 0; k < DIM; k++){
			 		  for (unsigned int l = 0; l < DIM; l++){
			 			  dWdF2[i][j][k][l]  =  ((i == k) && (j ==l) ? mu : 0.0)
			 					  + mu*Finv[j][k]*Finv[l][i]
								  + bulk*(2*J*J-J)*Finv[j][i]*Finv[l][k]
			 					  - bulk*(J*J-J)*Finv[j][k]*Finv[l][i];
			 		  }
		 		  }
	 		  }
	  	  }
  }

  void LinearElastic::get_F(Tensor<2, DIM> &grad_u, Tensor<2, DIM> &F){
	  for (unsigned int t = 0; t < DIM; t++){
	  		  for (unsigned int v = 0; v < DIM; v ++){
	  			  F[t][v] = (t==v) ? grad_u[t][v]+1.0 : grad_u[t][v];
	  		  }}
  }

  void LinearElastic::get_C(Tensor<2, DIM> &grad_u, Tensor<2, DIM> &C){

	  Tensor<2,DIM> F;

	 		  for (unsigned int t = 0; t < DIM; t++){
	 			  for (unsigned int v = 0; v < DIM; v ++){
	 				  F[t][v] = (t==v) ? grad_u[t][v]+1.0 : grad_u[t][v];
	 			  }}
	 C = contract<1,0>(transpose(F),F);
  }


  PowerLaw::PowerLaw() {}

  PowerLaw::~PowerLaw() {}

  void PowerLaw::Update(const double f0_, const double C1_, const double r0_,
    const double x_, const double C3_)
  {

    f0 = f0_;
    C1 = C1_;
    r0 = r0_;
    x = x_;
    C3 = C3_;
    xpp = x + 1.0;
    xmm = x - 1.0;
    C0 = (C1)*(r0) / xpp;
    C2 = (C1)*(x) / (r0);
    C4 = C0*pow(C3, xpp);
  }

  double
  PowerLaw::E(const double &r) const
  {
    if(r < 0.0)
       return 0.0;
     double s = (C3) + r / (r0);
     return (f0) * r - C4 + C0 * pow(s, xpp);
  }

  double
  PowerLaw::DE(const double &r) const
  {
    if(r < 0.0)
      return 0.0;
    double s = (C3) + r / (r0);
    return (f0) + (C1) * pow(s, x);
  }

  double
  PowerLaw::DDE(const double &r) const
  {
    if(r <= 0.0 && xmm < 0.0)
    {
//      std::cout << "BAD. " << r <<  " " <<   pow(r/r0,xmm) << std::endl;
      return 1.0e16;
    }
    else
    if(r < 0.0)
      return 0.0;
    double s = (C3) + r / (r0);
    return C2 * pow(s, xmm);
  }

#endif
