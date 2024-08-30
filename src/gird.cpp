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

#include "grid.h"

/**
 * @brief Constructs a Grid object with the specified minimum corner, maximum corner, and cell size.
 *
 * @param minCorner The minimum corner of the grid.
 * @param maxCorner The maximum corner of the grid.
 * @param h The cell size, equal to the smoothing length.
 */
Grid::Grid(float3_t minCorner, float3_t maxCorner, float_t h)
{
    // set min and max corners of the grid
    // add a small value to max and subtract a small value from min to avoid particles on the boundary
    this->minCorner = minCorner - 1.e-6;
    this->maxCorner = maxCorner + 1.e-6;

    dx = h;                    // set cell size equal to the smoothin length
    l = maxCorner - minCorner; // compute grid size

    // compute number of cells in each direction
    nx = ceil(l.x / dx);
    ny = ceil(l.y / dx);
    nz = ceil(l.z / dx);

    // compute total number of cells
    numCell = nx * ny * nz;

    // allocate memory for cellStart and cellEnd
    cellStart = new int[numCell];
    cellEnd = new int[numCell];

    // initialize cellStart with max value to indicate empty cells
    std::fill(cellStart, cellStart + numCell, std::numeric_limits<int>::max());
}

/**
 * Computes the hashes for the particles in the grid.
 *
 * This function calculates the hash value for each particle in the grid based on its position.
 * The hash value is used for spatial indexing and is computed using the formula:
 * hash = iz * ny * nx + iy * nx + ix
 *
 * @param particles A pointer to the Particles object containing the particle positions.
 */
void Grid::compute_hashes(Particles *particles)
{
    for (size_t i = 0; i < particles->N; i++)
    {
        // get the position of the particle
        float_t _px = particles->pos[i].x;
        float_t _py = particles->pos[i].y;
        float_t _pz = particles->pos[i].z;
        // compute the cell indices
        int ix = (_px - minCorner.x) / dx;
        int iy = (_py - minCorner.y) / dx;
        int iz = (_pz - minCorner.z) / dx;
        // compute the hash value
        particles->hash[i] = iz * ny * nx + iy * nx + ix;
    }
}

int Grid::compute_hash(int ix, int iy, int iz)
{
    return iz * ny * nx + iy * nx + ix;
}

/**
 * Computes the unhashed grid indices for a given hash value.
 *
 * @param hv The hash value.
 * @param ix Reference to the variable to store the computed x-index.
 * @param iy Reference to the variable to store the computed y-index.
 * @param iz Reference to the variable to store the computed z-index.
 */
void Grid::compute_unhashes(int hv, int &ix, int &iy, int &iz)
{
    ix = hv % nx;        // compute grid cell x-index
    iy = (hv / nx) % ny; // compute grid cell y-index
    iz = hv / (nx * ny); // compute grid cell z-index
}

/**
 * Sorts the particles based on their hash values.
 *
 * @param particles A pointer to the Particles object containing the particles to be sorted.
 */
void Grid::hash_sort(Particles *particles)
{
    std::vector<size_t> indices(particles->N);    // create a vector of indices
    std::iota(indices.begin(), indices.end(), 0); // fill the vector with increasing values starting from 0

    // sort the indices based on the hash values
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j)
              { return particles->hash[i] < particles->hash[j]; });

    // buffers to store sorted properties
    std::vector<float3_t> sortedPos(particles->N);
    std::vector<float_t> sortedT(particles->N);
    std::vector<float_t> sortedCp(particles->N);
    std::vector<float_t> sortedK(particles->N);
    std::vector<float_t> sortedRho(particles->N);
    std::vector<float_t> sortedH(particles->N);
    std::vector<float_t> sortedMass(particles->N);
    std::vector<int> sortedHash(particles->N);

    // sort the properties based on the sorted indices
    for (size_t i = 0; i < particles->N; i++)
    {
        size_t index = indices[i];
        sortedPos[i] = particles->pos[index];
        sortedT[i] = particles->T[index];
        sortedCp[i] = particles->cp[index];
        sortedK[i] = particles->k[index];
        sortedRho[i] = particles->rho[index];
        sortedH[i] = particles->h[index];
        sortedMass[i] = particles->mass[index];
        sortedHash[i] = particles->hash[index];
    }
    // copy the sorted properties back to the original arrays
    for (size_t i = 0; i < particles->N; i++)
    {
        particles->pos[i] = sortedPos[i];
        particles->T[i] = sortedT[i];
        particles->cp[i] = sortedCp[i];
        particles->k[i] = sortedK[i];
        particles->rho[i] = sortedRho[i];
        particles->h[i] = sortedH[i];
        particles->mass[i] = sortedMass[i];
        particles->hash[i] = sortedHash[i];
    }
}

/**
 * Computes the start and end indices of each cell in the grid based on the hash values of the particles.
 *
 * @param particles A pointer to the Particles object containing the hash values of the particles.
 */
void Grid::compute_cellStartEnd(Particles *particles)
{
    // get the first hash value (first cell)
    int currentHash = particles->hash[0];
    // set the start index of the first cell
    cellStart[currentHash] = 0;

    // iterate over the particles to compute the start and end indices of each cell
    for (size_t i = 1; i < particles->N; i++)
    {
        // if the hash value of the current particle is different from the previous one
        if (particles->hash[i] != currentHash)
        {
            // set the end index of the previous cell
            cellEnd[currentHash] = i - 1;
            // update the current hash value
            currentHash = particles->hash[i];
            // set the start index of the current cell
            cellStart[currentHash] = i;
        }
    }
    // set the end index of the last cell
    cellEnd[currentHash] = particles->N - 1;
}

long int Grid::get_nx() const
{
    return nx;
}

long int Grid::get_ny() const
{
    return ny;
}

long int Grid::get_nz() const
{
    return nz;
}

long int Grid::get_numCell() const
{
    return numCell;
}

float3_t Grid::get_minCorner() const
{
    return minCorner;
}

float3_t Grid::get_maxCorner() const
{
    return maxCorner;
}