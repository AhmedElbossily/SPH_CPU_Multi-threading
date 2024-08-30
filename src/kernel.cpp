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

#include "kernel.h"
float_t lapl_pse(float3_t posi, float3_t posj, float_t hi)
{
    // corrdinates of particle i
    float_t xi = posi.x;
    float_t yi = posi.y;
    float_t zi = posi.z;

    // corrdinates of particle j
    float_t xj = posj.x;
    float_t yj = posj.y;
    float_t zj = posj.z;

    // distance between particle i and j
    float_t xij = xi - xj;
    float_t yij = yi - yj;
    float_t zij = zi - zj;
    float_t xx = sqrt_t(xij * xij + yij * yij + zij * zij);

    // Too close, or out of concern
    if (xx < 1.e-6 || xx > 2 * hi)
        return 0;

    // calculate weight
    float_t h2 = hi * hi;
    float_t h4 = h2 * h2;
    float_t h5 = hi * h4;
    float_t w2_pse = +4. / (h5 * M_PI * sqrt(M_PI)) * exp(-xx * xx / (h2));

    return w2_pse;
}


// This function calculates the cubic spline weight and its derivatives between two particles.
// It takes the position of particle i (posi), the position of particle j (posj), and the smoothing length (hi) as inputs.
// It returns a float4_t variable (w) that contains the weight and its derivatives.

float4_t cubic_spline(float3_t posi, float3_t posj, float_t hi) {
    float4_t w = {0.0, 0.0, 0.0, 0.0};

    // Calculate the difference between the positions of particle i and particle j
    float3_t diff = {posi.x - posj.x, posi.y - posj.y, posi.z - posj.z};
    float_t rr2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    float_t h1 = 1 / hi;
    float_t fourh2 = 4 * hi * hi;

    // Check if the particles are too far apart or too close
    if (rr2 >= fourh2 || rr2 < 1e-8) {
        return w;
    }

    float_t rad = sqrtf(rr2);
    float_t q = rad * h1;
    float_t fac = (M_1_PI) * h1 * h1 * h1;

    // Calculate the weight and its derivatives based on the value of q
    if (q > 1) {
        float_t _2mq = 2 - q;
        float_t _2mq2 = _2mq * _2mq;
        float_t val = 0.25 * _2mq2 * _2mq;
        float_t der = -0.75 * _2mq2 * h1 / rad;

        // Assign the calculated values to the components of the float4_t variable w
        w.x = val * fac;         // weight
        w.y = der * diff.x * fac;    // derivative with respect to x
        w.z = der * diff.y * fac;    // derivative with respect to y
        w.w = der * diff.z * fac;    // derivative with respect to z
    } else {
        float_t val = 1 - 1.5 * q * q * (1 - 0.5 * q);
        float_t der = -3.0 * q * (1 - 0.75 * q) * h1 / rad;

        // Assign the calculated values to the components of the float4_t variable w
        w.x = val * fac;         // weight
        w.y = der * diff.x * fac;    // derivative with respect to x
        w.z = der * diff.y * fac;    // derivative with respect to y
        w.w = der * diff.z * fac;    // derivative with respect to z
    }

    return w;
}
