// SPH_thermal_CPU - Copyright Ahmed Elbossily

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.

// SPH_thermal_CPU is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License along with this program.
// If not, see <http://www.gnu.org/licenses/>.

#ifndef GRID
#define GRID

#include "type.h"
#include <vector>
#include <particle.h>
#include <algorithm>
#include <numeric>

/**
 * @class Grid
 * @brief Represents a grid used for spatial hashing.
 *
 * The Grid class provides methods to compute hash values for particles,
 * sort particles based on hash values, and compute cell start and end indices
 * for particles in the grid.
 */
class Grid
{
public:
	// Constructor and destructor
	Grid(float3_t minCorner, float3_t maxCornerInit, float_t h);
	~Grid() {};

	// cell start and end indices arrays
	int *cellStart = 0;
	int *cellEnd = 0;

	// helper methods
	void compute_hashes(Particles *particles);			   // compute hash values for all particles
	int compute_hash(int i, int j, int k);				   // compute hash values for a cell
	void compute_unhashes(int hv, int &i, int &j, int &k); // unhash a hash value to get the cell indices
	void hash_sort(Particles *particles);				   // sort particles based on hash values
	void compute_cellStartEnd(Particles *particles);	   // compute cell start and end indices for particles

	// Getter methods
	long int get_nx() const;
	long int get_ny() const;
	long int get_nz() const;
	long int get_numCell() const;
	float3_t get_minCorner() const;
	float3_t get_maxCorner() const;

private:
	float_t dx;						   // cell size
	float3_t l;						   // grid size
	long int nx = 0, ny = 0, nz = 0;   // number of cells in each direction
	long int numCell = 0;			   // total number of cells
	float3_t minCorner, maxCorner; // min and max corners of the grid
};

#endif /* Grid */
