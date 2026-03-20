/*
    tetmag - A general-purpose finite-element micromagnetic simulation software package
    Copyright (C) 2016-2023 CNRS and Université de Strasbourg

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
 * VTKwriter.h
 *
 *  Created on: Oct 10, 2016
 *      Author: riccardo
 */

#ifndef MESHWRITER_H_
#define MESHWRITER_H_
#include <Eigen/Dense>
#include <string>
#include <utility>
#include <vector>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
class MeshWriter {
private:
	Eigen::MatrixXi fel;
	Eigen::MatrixXd xyz;
	std::string name;
	unsigned ntet;
	unsigned nx;
	Eigen::VectorXi NodeMaterial;
	std::string sequenceFileName;
	std::vector<std::pair<double, std::string>> pvdEntries;
	vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid;
	void defineUnstructuredVTKGrid();
	void VTKwrite(int, Eigen::MatrixXd&, double, std::string);
	void writePVD();
	vtkSmartPointer<vtkDoubleArray> setFieldVTK(const Eigen::MatrixXd&, std::string);
	vtkSmartPointer<vtkIntArray> setMaterialVTK(const Eigen::VectorXi& );
	vtkSmartPointer<vtkDoubleArray> setScalarFieldVTK(const Eigen::VectorXd& , std::string );
public:
	void graphicsOutput(int, Eigen::MatrixXd&, double real_time = 0, std::string = "timeInPs");
	void setSequenceFileName(int);
	void outputVTK(std::string, const Eigen::MatrixXd&, double real_time = 0, std::string = "timeInPs");
	void addVectorVTK(const Eigen::MatrixXd&, std::string);
	void addScalarVTK(const Eigen::VectorXd&, std::string);
	void closeVTK();
	void writeBoundaryVTK(const Eigen::MatrixXi&, const Eigen::MatrixXd&);
//	void outputVTPolydata(std::string, const Eigen::MatrixXd&, double real_time = 0);
	MeshWriter(){};
	MeshWriter(const Eigen::MatrixXi&, const Eigen::MatrixXd&, std::string, Eigen::VectorXi);
};

#endif /* MESHWRITER_H_ */
