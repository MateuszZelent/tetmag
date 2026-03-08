/*
    tetmag - A general-purpose finite-element micromagnetic simulation software package
    Copyright (C) 2016-2026 CNRS and Université de Strasbourg

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
 * GMSHReader.cpp
 *
 *  Created on: Feb 1, 2019
 *      Author: Riccardo Hertel
 *
 */
#include <Eigen/Dense>
#include <string>
#include <mshio/mshio.h>
#include <vector>
#include <map>
#include <iostream>
#include "GMSHReader.h"
#include <numeric>
using namespace Eigen;
using RowMatrixXd = Matrix<double, Dynamic, Dynamic, RowMajor>;
using RowMatrixXi = Matrix<int, Dynamic, Dynamic, RowMajor>;

GMSHReader::GMSHReader(std::string name) : filename(name + ".msh") {
  mshio::MshSpec spec = mshio::load_msh(filename);

  std::vector<double> coords;
  for (const mshio::NodeBlock& block : spec.nodes.entity_blocks) {
    for (size_t i = 0; i < block.num_nodes_in_block; ++i) {
      nodeTags.push_back(block.tags[i]);
      coords.push_back(block.data[3 * i]);
      coords.push_back(block.data[3 * i + 1]);
      coords.push_back(block.data[3 * i + 2]);
    }
  }
  nx = nodeTags.size();
  xyz.resize(nx, 3);
  xyz = Map<RowMatrixXd>(coords.data(), nx, 3);

  const int tetElementType = 4;
  const int nodesPerTet = 4;
  std::vector<int> connectivity;
  nel = 0;
  for (const mshio::ElementBlock& block : spec.elements.entity_blocks) {
    if (block.element_type != tetElementType) continue;
    size_t stride = nodesPerTet + 1;
    for (size_t i = 0; i < block.num_elements_in_block; ++i) {
      for (int j = 1; j <= nodesPerTet; ++j) {
        connectivity.push_back(static_cast<int>(block.data[i * stride + j]));
      }
      ++nel;
    }
  }
  fel.resize(nel, 4);
  fel = Map<RowMatrixXi>(connectivity.data(), nel, 4);
}


void GMSHReader::readPoints() {}


void GMSHReader::readCells() {}


bool GMSHReader::renumberNodes(){
  bool hasFailed = false;
  std::map<int,int> nodeTagsMap;
  for (size_t i = 0; i < nx; ++i) {
    nodeTagsMap.insert(std::make_pair(nodeTags[i],i));
  }
  for (size_t i = 0; i < nel; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      std::map<int,int>::iterator posKey = nodeTagsMap.find(fel(i,j));
      if (posKey != nodeTagsMap.end()) {
	fel(i,j) = posKey->second;
      } else { hasFailed = true; }
    }
  }
  return hasFailed;
}


void GMSHReader::clean() {
  renumberNodes();
  if ( hasOrphanNodes() ) {
    deleteOrphanNodes();
    renumberNodes();
  }
};


int GMSHReader::hasOrphanNodes() {
  isOrphan.resize( nx );
  isOrphan.assign( nx, true );
  for (size_t i = 0; i < nel; ++i) {
    for (int j = 0; j < 4; ++j) {
      isOrphan[fel(i,j)] = false;
    }
  }
  totalOrphans = std::accumulate(isOrphan.begin(), isOrphan.end(), 0);
  return totalOrphans;
}


void GMSHReader::deleteOrphanNodes() {
  MatrixXd xyz_new(nx - totalOrphans, 3);
  std::cout << "Removing " << totalOrphans << " orphan node" << (totalOrphans >1 ? "s" : "") << std::endl;
  int reindex = 0;
  nodeTags.resize(nx - totalOrphans);
  for (size_t i = 0; i < nx; ++i) {
    if (!isOrphan[i]) {
      xyz_new.row(reindex) = xyz.row(i);
      nodeTags[reindex] = i;
      ++reindex;
    }
  }
  xyz = xyz_new;
};


MatrixXi GMSHReader::getElements() {
  return fel;
}


MatrixXd GMSHReader::getNodes() {
  return xyz;
}


std::vector<size_t> GMSHReader::getNodeTags() {
  return nodeTags;
}
