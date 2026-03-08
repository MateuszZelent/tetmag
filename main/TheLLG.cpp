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
 * TheLLG.cpp
 *
 *  Created on: May 10, 2017
 *      Author: riccardo
 */
///////////////

#include "TheLLG.h"
#include "auxiliaries.h"
#include "PhysicalConstants.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Eigen;

static double odeTime;
enum Coords
{
	x,
	y,
	z
};

// constructor
TheLLG::TheLLG(SimulationData &sd, const MeshData &msh, int LLGVersion) : nx(sd.nx),
																		  NodeVolume(msh.NodeVolume), alpha(sd.alpha), Js(sd.Js),
																		  hp(sd.Hp), hl(sd.Hl),
																		  freezeDemag(true)
{
	useGPU = sd.useGPU;
#ifdef USE_CUDA
	if (useGPU)
	{
		gpucalc = std::make_shared<EffFieldGPU>(nx);
	}
#endif
	efc = std::make_shared<EffFieldCalc>(sd, msh);
	odeTime = 0.;
	refresh_time = 0.1; // ToDo: consider replacing hard-coded value with user input dat
	heffTimer.reset();
	hl.setValueFunction();
	selectLLGType(LLGVersion);
	invNodeVol = NodeVolume.cwiseInverse();
	Hku = MatrixXd::Zero(nx, 3);
	Hsurf = MatrixXd::Zero(nx, 3);
	Hexc = MatrixXd::Zero(nx, 3);
	Hcub = MatrixXd::Zero(nx, 3);
	Heff = MatrixXd::Zero(nx, 3);
	Hdmi = MatrixXd::Zero(nx, 3);
	Hpls = MatrixXd::Zero(nx, 3);
	Hswp = MatrixXd::Zero(nx, 3);
	Hdem = MatrixXd::Zero(nx, 3);
	Hloc = MatrixXd::Zero(nx, 3);
	ret = MatrixXd::Zero(nx, 3);
	ret_vec.resize(ret.size());
	useDMI = !sd.D.isZero();
	if (useGPU)
	{
#ifdef USE_CUDA
		SpMat XC_field_OP = (-2.) * msh.stiff;
		XC_field_OP = (sd.A.cwiseProduct(invNodeVol)).asDiagonal() * XC_field_OP;
		std::cout << "setting exchange matrix on device" << std::endl;
		gpucalc->setExchangeMatOnDev(XC_field_OP);
		if (useDMI)
		{
			std::cout << "setting gradient matrices on device" << std::endl;
			gpucalc->setGradientMatsOnDev(-msh.tGradX, -msh.tGradY, -msh.tGradZ);
			gpucalc->setDMIdata(sd.D, invNodeVol, msh.nv_nx, msh.nodeArea);
		}
#endif
	}
	useCubic = !(sd.Kc1.isZero() && sd.Kc2.isZero());
	useUniaxial = !sd.Ku1.isZero();
	useSurfaceAnisotropy = !sd.Ks.isZero();
	if (useUniaxial && useGPU)
	{
#ifdef USE_CUDA
		gpucalc->setUniaxialAnisotropy(sd.Kuni, sd.Ku1);
#endif
	}
	realTimeScale = sd.psTimeScaleFactor();
	invJs = Js.cwiseInverse().unaryExpr([](double v)
										{ return std::isfinite(v) ? v : 0.0; });
	JsVol = Js.cwiseProduct(NodeVolume);
	if (hp.pulseIsUsed || hp.sweepIsUsed)
		setHdynField();
	totalVolume = NodeVolume.sum();
	selectEffectiveFields();
}

///////////////////////////// OPERATOR () FOR ODEINT AND CVODE /////////////////////////////////

void TheLLG::operator()(const state_type &state, state_type &dxdt, const double theTime)
{
	timeInPs = theTime / realTimeScale;
	odeTime = timeInPs;
	dxdt = selectedLLGType(state);
}

//////////////////////////// LLG TYPE SELECTION ///////////////////////////////////

void TheLLG::selectLLGType(int choice)
{
	enum choices
	{
		noPrec,
		usualLLG,
		STT
	};
	useSTT = false;
	if (useGPU)
	{
#ifdef USE_CUDA
		selectLLGTypeGPU(choice);
#endif
	}
	if (choice == noPrec)
	{
		selectedLLGType = [this](const state_type &m) -> state_type
		{ return noPrecession(m); };
	}
	else if (choice == STT)
	{
		useSTT = true;
		selectedLLGType = [this](const state_type &m) -> state_type
		{ return sttDynamics(m); };
	}
	else
	{
		selectedLLGType = [this](const state_type &m) -> state_type
		{ return classicVersion(m); };
	}
}

/////////////////////////// LLG VERSIONS //////////////////////////

state_type TheLLG::classicVersion(const state_type &mag_vec)
{
	if (useGPU)
	{
#ifdef USE_CUDA
		gpucalc->setMagDev(mag_vec);
#endif
	} // requires ColMajor storage for Mag
	Map<const MatrixXd_CM> Mag(mag_vec.data(), nx, 3);
	evaluateAllEffectiveFields(Mag);
	Heff = totalEffectiveField();
	if (useGPU)
	{
#ifdef USE_CUDA
		// This is invoked in the option "integrate LLG on CPU while using GPU for eff. field calculation"
		ret_vec = gpucalc->ClassicLLG_hst(Heff, alpha);
#endif
	}
	else
	{
		const MatrixXd MxH = cross(Mag, Heff);
		Map<MatrixXd_CM>(ret_vec.data(), ret.rows(), 3) = -MxH - alpha * cross(Mag, MxH);
	}
	return (ret_vec);
}

state_type TheLLG::noPrecession(const state_type &mag_vec)
{
	if (useGPU)
	{
#ifdef USE_CUDA
		gpucalc->setMagDev(mag_vec);
#endif
	}
	Map<const MatrixXd_CM> Mag(mag_vec.data(), nx, 3);
	evaluateAllEffectiveFields(Mag);
	Heff = totalEffectiveField();
	if (useGPU)
	{
#ifdef USE_CUDA
		ret_vec = gpucalc->LLG_noPrec_hst(Heff, alpha);
#endif
	}
	else
	{
		Map<MatrixXd_CM>(ret_vec.data(), ret.rows(), 3) = -alpha * cross(Mag, cross(Mag, Heff));
	}
	return (ret_vec);
}

state_type TheLLG::sttDynamics(const state_type &mag_vec)
{
	const std::vector<double> LLGpart = classicVersion(mag_vec);
	Map<const MatrixXd_CM> Mag(mag_vec.data(), nx, 3);
	calcUtermSTT(Mag);
	if (useGPU)
	{
#ifdef USE_CUDA
		ret_vec = gpucalc->STT_term_LLG_hst(stt.Ustt, alpha, stt.beta);
#endif
	}
	else
	{
		const MatrixXd MxU = cross(Mag, stt.Ustt);
		ret = -(stt.beta - alpha) * MxU - (1. + alpha * stt.beta) * cross(Mag, MxU);
		Map<MatrixXd_CM>(ret_vec.data(), ret.rows(), 3) = ret;
	}
	if (stt.pulseIsUsed)
	{
		double pulseVal = stt.gaussPulseValue(timeInPs);
		std::transform(ret_vec.begin(), ret_vec.end(), ret_vec.begin(), [pulseVal](double &c)
					   { return c * pulseVal; });
	}
	std::transform(ret_vec.begin(), ret_vec.end(), LLGpart.begin(), ret_vec.begin(), std::plus<double>());
	return (ret_vec);
}

////////////////////// EFFECTIVE FIELDS ////////////////////////////

void TheLLG::selectEffectiveFields()
{
	calcEffectiveField.clear();
	calcEffectiveField.emplace_back([this](MRef &m)
									{ calcExchangeField(m); });
	if (useUniaxial)
		calcEffectiveField.emplace_back([this](MRef &m)
										{ calcUniaxialAnisotropyField(m); });
	if (useCubic)
		calcEffectiveField.emplace_back([this](MRef &m)
										{ calcCubicAnisotropyField(m); });
	if (useDMI)
		calcEffectiveField.emplace_back([this](MRef &m)
										{ calcDMIField(m); });
	if (useSurfaceAnisotropy)
		calcEffectiveField.emplace_back([this](MRef &m)
										{ calcSurfaceAnisotropyField(m); });
	if (hp.pulseIsUsed)
		calcEffectiveField.emplace_back([this](MRef &m)
										{ calcPulseField(m); });
	if (hp.sweepIsUsed)
		calcEffectiveField.emplace_back([this](MRef &m)
										{ calcSweepField(m); });
	if (hl.isUsed)
		calcEffectiveField.emplace_back([this](MRef &m)
										{ calcLocalField(m); });
}

MatrixXd TheLLG::totalEffectiveField()
{
	MatrixXd Htot(nx, 3);
#ifdef _OPENMP
	int i, j;
#pragma omp parallel for private(i) collapse(2)
	for (i = 0; i < nx; ++i)
	{
		for (j = 0; j < 3; ++j)
		{
			Htot(i, j) = (Hexc(i, j) + Hku(i, j) + Hcub(i, j) + Hsurf(i, j) + Hdmi(i, j)) * invJs(i) + Hext(i, j) + Hdem(i, j) + Hpls(i, j) + Hswp(i, j) + Hloc(i, j);
		}
	}
#else
	MatrixXd Heff_tot = (Hexc + Hku + Hcub + Hsurf + Hdmi).array().colwise() * invJs.array();
	Htot = Heff_tot + Hext + Hdem + Hpls + Hswp + Hloc;
#endif
	return Htot;
}

void TheLLG::evaluateAllEffectiveFields(MRef &Mag)
{
	int tasks = calcEffectiveField.size();
	for (int i = 0; i < tasks; ++i)
	{
		calcEffectiveField[i](Mag);
	}
}

void TheLLG::calcExchangeField(MRef &Mag)
{
	heffTimer.start();
	if (useGPU)
	{
#ifdef USE_CUDA
		Hexc = gpucalc->ExchangeFieldGPU();
#endif
	}
	else
	{
		Hexc = efc->exchangeField(Mag);
	}
	heffTimer.add();
}

void TheLLG::calcDemagField(MRef &Mag)
{
	if (odeTime >= next_refresh)
	{
		next_refresh += refresh_time;
		Hdem = demag->calcField(Mag);
	}
}

void TheLLG::calcUniaxialAnisotropyField(MRef &Mag)
{
	if (useGPU)
	{
#ifdef USE_CUDA
		Hku = gpucalc->UniaxialAnisotropyField();
#endif
	}
	else
	{
		Hku = efc->uniaxialAnisotropyField(Mag);
	}
}

void TheLLG::calcCubicAnisotropyField(MRef &Mag)
{
	if (useCubic)
	{
		Hcub = efc->cubicAnisotropyField(Mag);
	}
}

void TheLLG::calcSurfaceAnisotropyField(MRef &Mag)
{
	Hsurf = efc->surfaceAnisotropyField(Mag);
}

void TheLLG::calcDMIField(MRef &Mag)
{
	if (useGPU)
	{
#ifdef USE_CUDA
		Hdmi = gpucalc->DMIField();
#endif
	}
	else
	{
		Hdmi = efc->dmiField(Mag);
	}
}

void TheLLG::calcPulseField(MRef &)
{
	Hpls = hp.pulseH * hp.gaussPulseValue(timeInPs);
}

void TheLLG::calcSweepField(MRef &)
{
	Hswp = hp.sweepH * hp.sweepFieldValue(timeInPs);
}

void TheLLG::calcLocalField(MRef &)
{
	Hloc = hl.localField(timeInPs);
}

///////////////////////  ENERGIES  ///////////////////////

double TheLLG::getExchangeEnergy(MRef &Mag)
{
	return efc->calcExchangeEnergy(Mag);
}
double TheLLG::getDirectExch(MRef &Mag)
{
	return efc->exchEnergy_direct(Mag);
}

double TheLLG::getZeemanEnergy(MRef &Mag)
{
	double zeeEnergy = -(JsVol.transpose() * (Hext + Hpls + Hswp + Hloc).cwiseProduct(Mag)).sum();
	return zeeEnergy;
}

double TheLLG::getDemagEnergy(MRef &Mag)
{
	if (freezeDemag)
		calcDemagField(Mag);
	return demag->getDemagEnergy(Mag);
}

double TheLLG::getUniaxialAnisotropyEnergy(MRef &Mag)
{
	if (!useUniaxial)
		return 0;
	return efc->uniaxialEnergy(Mag);
}

double TheLLG::getCubicAnisotropyEnergy(MRef &Mag)
{
	if (!useCubic)
		return 0;
	return efc->cubicAnisotropyEnergy(Mag);
}

double TheLLG::getDMIEnergy(MRef &Mag)
{
	if (!useDMI)
		return 0;
#ifdef USE_CUDA
	calcDMIField(Mag);
	efc->setDMIField(Hdmi);
#endif
	return efc->dmiEnergy(Mag);
}

double TheLLG::getSurfaceAnisotropyEnergy(MRef &Mag)
{
	if (!useSurfaceAnisotropy)
		return 0.;
	return efc->surfaceAnisotropyEnergy(Mag);
}

double TheLLG::getDirectDMI(MRef &Mag)
{
	return efc->dmiEnergy_direct(Mag);
}

/////////////////////////// OTHER /////////////////////////////////////////////

void TheLLG::outputTimer()
{
	std::cout << "time for copying data [s]:\t" << std::setprecision(2) << copyTimer.durationInMus() / 1.e6 << std::endl;
#ifdef USE_CUDA
	gpucalc->displayTimer();
#endif
	std::cout << "time for effective field: " << std::setprecision(2) << heffTimer.durationInMus() / 1.e6 << std::endl;
	std::cout << std::setprecision(-1);
}

void TheLLG::calcUtermSTT(MRef &Mag)
{
	if (useGPU)
	{
#ifdef USE_CUDA
		stt.Ustt = gpucalc->UTermSTT_GPU();
#endif
	}
	else
	{
		// #pragma omp parallel for // SparseMVP is already parallelized by Eigen.
		for (int i = 0; i < 3; ++i)
		{
			stt.Ustt.col(i) = stt.eta_jx.cwiseProduct(stt.gradX * Mag.col(i)) + stt.eta_jy.cwiseProduct(stt.gradY * Mag.col(i)) + stt.eta_jz.cwiseProduct(stt.gradZ * Mag.col(i));
		}
	}
}

/////////////////// GETTER  //////////////////////////

Vector3d TheLLG::getMeanH()
{
	return NodeVolume.transpose() * (Hext + Hpls + Hswp + Hloc) / totalVolume;
}

double TheLLG::getMaxTorque(MRef &Mag)
{
	evaluateAllEffectiveFields(Mag);
	Heff = totalEffectiveField();
	double m_torque = 0;
	if (useGPU)
	{
#ifdef USE_CUDA
		m_torque = gpucalc->MaxTorque(Heff);
#endif
	}
	else
	{
		m_torque = cross(Mag, Heff).rowwise().norm().maxCoeff();
	}
	// Note: The max.-torque criterion only includes effective field contributions (no STT torques).
	return m_torque * PhysicalConstants::mu0;
}

MatrixXd fieldFromAngles(int nx, double Habs, double theta_h, double phi_h)
{
	MatrixXd Hfield = MatrixXd::Ones(nx, 3); // use other distribution for possible extension of inhomogeneous field
	Hfield.col(x) *= std::sin(theta_h) * std::cos(phi_h);
	Hfield.col(y) *= std::sin(theta_h) * std::sin(phi_h);
	Hfield.col(z) *= std::cos(theta_h);
	Hfield *= (Habs / PhysicalConstants::mu0);
	return Hfield;
}

/////////////////////////////////// SETTER ///////////////////////////////////////////

void TheLLG::setTime(double time)
{
	timeInPs = time;
}

void TheLLG::setDemagData(const DemagField &demag_)
{
	demag = std::make_shared<DemagField>(demag_);
	freezeDemag = false;
	calcEffectiveField.emplace_back([this](MRef &m)
									{ calcDemagField(m); });
}

void TheLLG::setSTTData(const STT &stt_)
{
	stt = stt_;
	stt.setEta(invJs);
	if (useGPU)
	{
#ifdef USE_CUDA
		std::cout << "setting STT data on device" << std::endl;
		gpucalc->setSTTDataOnDevice(stt.gradX, stt.gradY, stt.gradZ, stt.eta_jx, stt.eta_jy, stt.eta_jz);
#endif
	}
}

void TheLLG::setZeemanField(int nx, double H_stat, double theta_H_deg, double phi_H_deg,
							double H_hys, double theta_Hys_deg, double phi_Hys_deg)
{
	double deg_to_rad = PhysicalConstants::pi / 180.;
	MatrixXd Hhys = fieldFromAngles(nx, H_hys, theta_Hys_deg * deg_to_rad, phi_Hys_deg * deg_to_rad);
	MatrixXd Hstat = fieldFromAngles(nx, H_stat, theta_H_deg * deg_to_rad, phi_H_deg * deg_to_rad);
	Hext = Hhys + Hstat;
}

void TheLLG::setHdynField()
{
	hp.pulseH = MatrixXd::Zero(nx, 3);
	hp.sweepH = MatrixXd::Zero(nx, 3);
	if (hp.pulseIsUsed)
	{
		hp.pulseH = fieldFromAngles(nx, 1., hp.pulseTheta, hp.pulsePhi);
	}
	if (hp.sweepIsUsed)
	{
		hp.sweepH = fieldFromAngles(nx, 1., hp.sweepTheta, hp.sweepPhi);
	}
}

void TheLLG::setMag(MRef &mag_)
{
	(void)mag_;
	if (useGPU)
	{
#ifdef USE_CUDA
		gpucalc->setMagDev(mag_);
#endif
	}
}

void TheLLG::setHdem(MRef &Hdem_)
{
	Hdem = Hdem_;
}

//////////////////////////////////// SUNDIALS INTEGRATOR //////////////////////////////////////////////////////

int TheLLG::integrateSUNDIALS(std::vector<double> &mag_vec, double ode_start_t, double ode_end_t, double dt)
{
	//	(void) dt;
	copyTimer.start();

	int its = integrator_(mag_vec, ode_start_t, ode_end_t, dt);
	copyTimer.add();
	return its;
}

void TheLLG::initIntegrator()
{
#ifdef USE_CUDA
	if (useGPU)
	{
		gpu_integrator_.reset(new GPU_Integrator(this, nx));

		integrator_ = [this](std::vector<double> &mag_vec, double ode_start_t, double ode_end_t, double dt) -> int
		{
			return gpu_integrator_->integrateCVODE(mag_vec, ode_start_t, ode_end_t, dt);
		};
		return;
	}
#endif

	cpu_integrator_.reset(new CPU_Integrator(this, nx));

	integrator_ = [this](std::vector<double> &mag_vec, double ode_start_t, double ode_end_t, double dt) -> int
	{
		return cpu_integrator_->integrateCVODE(mag_vec, ode_start_t, ode_end_t, dt);
	};
}

std::shared_ptr<UserData> TheLLG::alloc_user_data(const int nx)
{
	std::shared_ptr<UserData> data = std::make_shared<UserData>();
	data->llg = this;
	data->nx = nx;
	data->ret.resize(3 * nx);
	data->mag.resize(3 * nx);
	return data;
}

int TheLLG::rhs(realtype t, N_Vector u, N_Vector u_dot, void *user_data)
{
	(void)t;
	UserData *u_data = static_cast<UserData *>(user_data);
	int n = 3 * u_data->nx;
	realtype *udata;

#ifdef _OPENMP
	udata = NV_DATA_OMP(u);
#else
	udata = N_VGetArrayPointer(u);
#endif

	std::vector<double> &mag = u_data->mag;
	std::copy(udata, udata + n, mag.begin());
	u_data->llg->operator()(mag, u_data->ret, t);

	realtype *dudata;
#ifdef _OPENMP
	dudata = NV_DATA_OMP(u_dot);
#else
	dudata = N_VGetArrayPointer(u_dot);
#endif
	std::copy(u_data->ret.begin(), u_data->ret.begin() + n, dudata);
	return (0); // <--- This is required to signal success
}

CPU_Integrator::CPU_Integrator(TheLLG *llg, int nx)
	: llg_(llg), nx_(nx), m_(NULL), cvode_mem_(NULL), LS_(NULL), data_(), sunctx_(NULL)
{
}

CPU_Integrator::~CPU_Integrator()
{
#ifdef _OPENMP
	if (m_ != NULL)
		N_VDestroy_OpenMP(m_);
#else
	if (m_ != NULL)
		N_VDestroy(m_);
#endif

	if (cvode_mem_ != NULL)
		CVodeFree(&cvode_mem_);
	if (LS_ != NULL)
		SUNLinSolFree(LS_);
	if (sunctx_ != NULL)
		SUNContext_Free(&sunctx_);
}

int CPU_Integrator::integrateCVODE(std::vector<double> &mag_vec, double ode_start_t, double ode_end_t, double dt)
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

	if (cvode_mem_ == NULL)
	{ // first call

		flag = SUNContext_Create(NULL, &sunctx_);
		if (flag != 0)
			return 1;

#ifdef _OPENMP
		int num_threads = omp_get_max_threads();
		m_ = N_VMake_OpenMP(N, mag_vec.data(), num_threads, sunctx_);
#else
		m_ = N_VMake_Serial(N, mag_vec.data(), sunctx_);
#endif
		if (m_ == NULL)
			return 1;
		cvode_mem_ = CVodeCreate(CV_ADAMS, sunctx_);
		if (cvode_mem_ == NULL)
			return 1;

		flag = CVodeInit(cvode_mem_, TheLLG::rhs, t0, m_);
		if (flag != CV_SUCCESS)
			return 1;

		flag = CVodeSStolerances(cvode_mem_, reltol, abstol);
		if (flag != CV_SUCCESS)
			return 1;

		data_ = llg_->alloc_user_data(nx_);
		flag = CVodeSetUserData(cvode_mem_, static_cast<void *>(data_.get()));
		if (flag != CV_SUCCESS)
			return 1;

		LS_ = SUNLinSol_SPGMR(m_, PREC_NONE, 0, sunctx_);
		if (LS_ == NULL)
			return 1;

		flag = CVodeSetLinearSolver(cvode_mem_, LS_, NULL);
		if (flag != CV_SUCCESS)
			return 1;
	}
	else
	{ // subsequent calls
		flag = CVodeReInit(cvode_mem_, t0, m_);
		if (flag != CV_SUCCESS)
			return 1;
	}

	CVodeGetNumNonlinSolvIters(cvode_mem_, &its_nl);
	CVodeGetNumLinIters(cvode_mem_, &its_l);

	flag = CVode(cvode_mem_, tout, m_, &t, CV_NORMAL);

	if (flag < 0)
		return 1;
	int its = static_cast<int>(its_nl + its_l);
	return its;
};
//////////////// END SUNDIALS //////////////////////////////
