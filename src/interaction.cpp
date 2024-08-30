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

#include "interaction.h"

/**
 * @brief Calculates the heat transfer between particles.
 *
 * This function calculates the heat transfer between particles using the SPH method.
 * The heat transfer is calculated based on the temperature difference between particles.
 *
 * @param particles A pointer to the Particles object.
 * @param grid A pointer to the Grid object.
 * @param start The starting index of the particles for this thread.
 * @param end The ending index of the particles for this thread.
 */
void interactions_heat(Particles *particles, Grid *grid, size_t start, size_t end)
{
    unsigned int N = particles->N;
    for (size_t i = start; i < end; i++)
    {
        float3_t pi = particles->pos[i];   // position of particle i
        float_t Ti = particles->T[i];      // temperature of particle i
        float_t cpi = particles->cp[i];    //  specific heat capacity of particle i
        float_t ki = particles->k[i];      // thermal conductivity of particle i
        float_t rhoi = particles->rho[i];  // density of particle i
        float_t hi = particles->h[i];      // smoothing length of particle i
        float_t mass = particles->mass[i]; // mass of particle
        float_t T_ti = 0.;                 // time derivative of temperature for particle i
        float_t hash = particles->hash[i]; // hash value of particle i

        // compute cell indices for particle i
        int ix, iy, iz;
        grid->compute_unhashes(hash, ix, iy, iz);

        // lowest indice of the neighboring cells
        // -1 to go one step back
        // adding condition to avoid negative indices
        // and indices greater than the grid size for boundary particles
        int low_i = ix - 1 < 0 ? 0 : ix - 1;
        int low_j = iy - 1 < 0 ? 0 : iy - 1;
        int low_k = iz - 1 < 0 ? 0 : iz - 1;

        // highest indice of the neighboring cells
        // +2 to go two steps forward
        // Why +2? Because we are not considering the last cell in the loop below
        int high_i = ix + 2 > grid->get_nx() ? grid->get_nx() : ix + 2;
        int high_j = iy + 2 > grid->get_ny() ? grid->get_ny() : iy + 2;
        int high_k = iz + 2 > grid->get_nz() ? grid->get_nz() : iz + 2;

        for (size_t ii = low_i; ii < high_i; ii++)
        {
            for (size_t jj = low_j; jj < high_j; jj++)
            {
                for (size_t kk = low_k; kk < high_k; kk++)
                {
                    // compute hash value for the neighboring cell
                    int h_idx = grid->compute_hash(ii, jj, kk);

                    // get the start and end indices of the particles in the neighboring cell
                    int start = grid->cellStart[h_idx];
                    int end = grid->cellEnd[h_idx];

                    // if the cell is empty, continue to the next cell
                    if (start == std::numeric_limits<int>::max())
                        continue;

                    for (size_t j = start; j <= end; j++)
                    {
                        float3_t pj = particles->pos[j];  // position of particle j
                        float_t Tj = particles->T[j];     // temperature of particle j
                        float_t cpi = particles->cp[j];   // specific heat capacity of particle j
                        float_t kj = particles->k[j];     // thermal conductivity of particle j
                        float_t rhoj = particles->rho[j]; // density of particle j

                        // distance between particle i and j
                        float_t xij = pi.x - pj.x;
                        float_t yij = pi.y - pj.y;
                        float_t zij = pi.z - pj.z;
                        float_t rij_2 = xij * xij + yij * yij + zij * zij;
                        float_t rij = sqrt(rij_2);

                        float4_t ww = cubic_spline(pi, pj, hi); // cubic spline kernel
                        //float_t w2_pse = lapl_pse(pi, pj, hi); // Laplacian by PSE-method

                        float_t w = ww.x;   // weight
                        float_t w_x = ww.y; // weight gradient
                        float_t w_y = ww.z; // weight gradient
                        float_t w_z = ww.w; // weight gradient 

                        // avoid division by zero
                        if (rij > 1e-8)
                        {
                            float_t eijx = xij / rij;
                            float_t eijy = yij / rij;
                            float_t eijz = zij / rij;
                            float_t rij1 = 1. / rij;

                            T_ti += ((4 * ki * kj) / (ki + kj)) * (mass / rhoj) * (Ti - Tj) * rij1 * (eijx * w_x + eijy * w_y + eijz * w_z);
                            // T_ti += (( ki * kj) / (ki + kj)) * (mass / rhoj) * (Tj - Ti) * w2_pse;
                           // T_ti += ((2 * ki * kj) / (ki + kj)) * (mass / rhoj) * (Tj - Ti) * w2_pse;
                        }
                    }
                }
            }
        }

        float_t alpha = 1.0 / (rhoi * cpi); //   thermal diffusivity of particle i
        particles->T_t[i] = alpha * T_ti;   // time derivative of temperature for particle i
    }
}
