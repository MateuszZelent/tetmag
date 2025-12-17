/*
	tetmag - A general-purpose finite-element micromagnetic simulation software package
	Copyright (C) 2016-2025 CNRS and Université de Strasbourg

	Author: Riccardo Hertel

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU Affero General Public License as
	published by the Free Software Foundation, either version 3 of the
	License, or (at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU Affero General Public License for more details.

	Contact: Riccardo Hertel, IPCMS Strasbourg, 23 rue du Loess,
					 67034 Strasbourg, France.
					 riccardo.hertel@ipcms.unistra.fr

	You should have received a copy of the GNU Affero General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * LLGWrapper.cu
 *
 *  Created on: Sep 23, 2019
 *      Author: riccardo
 */

// Cannot be included in TheLLG class: Separate translation unit needed for nvcc.

#include "LLGWrapper.h"
#include <functional>
#include <thrust/copy.h>
#include <nvector/nvector_cuda.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>

dev_vec mag_vec_tmp;
dev_vec ret_vec_tmp;

// using namespace boost::numeric::odeint;
// typedef runge_kutta_fehlberg78<dev_vec, double, dev_vec, double> fehlberg78_gpu;
// typedef controlled_runge_kutta< fehlberg78_gpu > fehlberg78_controlled_gpu;

void LLGWrapper::init(const int nx)
{
	nx3 = 3 * nx;
	mag_vec_tmp.resize(nx3);
	ret_vec_tmp.resize(nx3);
}

dev_vec TheLLG::sttDynamics_GPU(const dev_vec &mag_vec)
{
	dev_vec LLGpart_d = classicVersion_GPU(mag_vec);
	stt.Ustt = gpucalc->UTermSTT_GPU();
	dev_vec ret_vec_d = *gpucalc->STT_term_LLG_dev(stt.Ustt, alpha, stt.beta);
	if (stt.pulseIsUsed)
	{
		double pulseVal = stt.gaussPulseValue(timeInPs);
		thrust::transform(ret_vec_d.begin(), ret_vec_d.end(), ret_vec_d.begin(), pulseVal * thrust::placeholders::_1);
	}
	thrust::transform(thrust::device, ret_vec_d.begin(), ret_vec_d.end(), LLGpart_d.begin(), ret_vec_d.begin(), thrust::plus<double>());
	return (ret_vec_d);
}

Eigen::MatrixXd TheLLG::effFieldsForGPU(const dev_vec &mag_vec)
{
	std::vector<double> mag_vec_h(3 * nx);
	thrust::copy(mag_vec.begin(), mag_vec.end(), mag_vec_h.begin());
	Eigen::Map<const MatrixXd_CM> Mag(mag_vec_h.data(), nx, 3);
	evaluateAllEffectiveFields(Mag);
	return totalEffectiveField();
}

dev_vec TheLLG::classicVersion_GPU(const dev_vec &mag_vec)
{
	gpucalc->setMagDev(mag_vec);
	Heff = effFieldsForGPU(mag_vec);
	return *gpucalc->ClassicLLG_dev(Heff, alpha);
}

dev_vec TheLLG::noPrecession_GPU(const dev_vec &mag_vec)
{
	gpucalc->setMagDev(mag_vec);
	Heff = effFieldsForGPU(mag_vec);
	return *gpucalc->LLG_noPrec_dev(Heff, alpha);
}

template <class deviceType>
void delete_vec(deviceType &x_d)
{
	x_d.clear();
	x_d.shrink_to_fit();
	x_d.~device_vector();
}

LLGWrapper::~LLGWrapper()
{
	cudaDeviceSynchronize();
	delete_vec(mag_vec_tmp);
	delete_vec(ret_vec_tmp);
}

void TheLLG::operator()(const thrust::device_vector<double> &mag_d, thrust::device_vector<double> &dxdt_d, const double theTime /*time*/)
{
	timeInPs = theTime / realTimeScale;
	dxdt_d = selectedLLGType_GPU(mag_d);
}

void TheLLG::selectLLGTypeGPU(int choice)
{
	enum choices
	{
		noPrec,
		usualLLG,
		STT
	};
	useSTT = false;
	if (choice == noPrec)
	{
		selectedLLGType_GPU = [this](const dev_vec &m) -> dev_vec
		{ return noPrecession_GPU(m); };
	}
	else if (choice == STT)
	{
		useSTT = true;
		selectedLLGType_GPU = [this](const dev_vec &m) -> dev_vec
		{ return sttDynamics_GPU(m); };
	}
	else
	{
		selectedLLGType_GPU = [this](const dev_vec &m) -> dev_vec
		{ return classicVersion_GPU(m); };
	}
}

int TheLLG::rhs_d(realtype t, N_Vector u, N_Vector u_dot, void *user_data)
{
	(void)t;
	UserData *u_data = static_cast<UserData *>(user_data);
	sunindextype N = 3 * u_data->nx;
	realtype *udata = N_VGetDeviceArrayPointer_Cuda(u);
	realtype *dudata = N_VGetDeviceArrayPointer_Cuda(u_dot);
	thrust::copy_n(udata, N, mag_vec_tmp.begin());
	u_data->llg->operator()(mag_vec_tmp, ret_vec_tmp, t);
	cudaMemcpy(dudata, thrust::raw_pointer_cast(ret_vec_tmp.data()), N * sizeof(realtype), cudaMemcpyDeviceToDevice);
	return 0; // <--- This is required to signal success
}

GPU_Integrator::~GPU_Integrator()
{
	if (m_gpu_ != NULL)
		N_VDestroy_Cuda(m_gpu_);
	if (cvode_mem_ != NULL)
		CVodeFree(&cvode_mem_);
	if (LS_ != NULL)
		SUNLinSolFree(LS_);
	if (sunctx_ != NULL)
		SUNContext_Free(&sunctx_);
}

GPU_Integrator::GPU_Integrator(TheLLG *llg, int nx)
	: llg_(llg), nx_(nx), m_gpu_(NULL), cvode_mem_(NULL), LS_(NULL), data_(), sunctx_(NULL) {}

int GPU_Integrator::integrateCVODE(std::vector<double> &mag_vec, double ode_start_t, double ode_end_t, double dt)
{
	(void)dt;

	long int its_l = 0;
	long int its_nl = 0;
	int flag;
	sunindextype N = static_cast<sunindextype>(3 * nx_);

	realtype t0 = ode_start_t;
	realtype t = ode_start_t;
	realtype reltol = 1.0e-6;
	realtype abstol = 1.0e-6;
	realtype tout = ode_start_t + ode_end_t;

	mag_vec_tmp = mag_vec; // host -> device copy

	if (cvode_mem_ == NULL)
	{
		flag = SUNContext_Create(NULL, &sunctx_);
		if (flag != 0)
			return 1;
		m_gpu_ = N_VMake_Cuda(N, mag_vec.data(), thrust::raw_pointer_cast(mag_vec_tmp.data()), sunctx_);
		if (m_gpu_ == NULL)
			return 1;

		cvode_mem_ = CVodeCreate(CV_ADAMS, sunctx_);
		if (cvode_mem_ == NULL)
			return 1;

		flag = CVodeInit(cvode_mem_, TheLLG::rhs_d, t0, m_gpu_);
		if (flag != CV_SUCCESS)
			return 1;

		flag = CVodeSStolerances(cvode_mem_, reltol, abstol);
		if (flag != CV_SUCCESS)
			return 1;

		data_ = llg_->alloc_user_data(nx_);
		flag = CVodeSetUserData(cvode_mem_, static_cast<void *>(data_.get()));
		if (flag != CV_SUCCESS)
			return 1;

		LS_ = SUNLinSol_SPGMR(m_gpu_, PREC_NONE, 0, sunctx_);
		if (LS_ == NULL)
			return 1;

		flag = CVodeSetLinearSolver(cvode_mem_, LS_, NULL);
		if (flag != CV_SUCCESS)
			return 1;
	}
	else
	{
		flag = CVodeReInit(cvode_mem_, t0, m_gpu_);
		if (flag != CV_SUCCESS)
			return 1;
	}

	// Integrate on GPU
	t = t0;
	flag = CVode(cvode_mem_, tout, m_gpu_, &t, CV_NORMAL);
	if (flag < 0)
	{
		std::cerr << "Warning: GPU integration failed, flag = " << flag << std::endl;
		return 1;
	}

	CVodeGetNumNonlinSolvIters(cvode_mem_, &its_nl);
	CVodeGetNumLinIters(cvode_mem_, &its_l);

	double *res_p = N_VGetDeviceArrayPointer_Cuda(m_gpu_);

	thrust::copy(res_p, res_p + N, mag_vec_tmp.begin());
	thrust::copy(mag_vec_tmp.begin(), mag_vec_tmp.end(), mag_vec.begin());
	llg_->gpucalc->setMagDev(mag_vec_tmp);
	int its = static_cast<int>(its_nl + its_l);
	return its;
}
