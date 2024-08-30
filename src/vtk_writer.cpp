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

#include "vtk_writer.h"
void vtk_writer_write(Particles *particles, unsigned int step)
{
    // number of particles in the simulation
    unsigned int N_particles = particles->N;

    char buf[256];
    // output file name
    sprintf(buf, "../results/output_%012d.vtk", step);
    FILE *fp = fopen(buf, "w+");

    // write vtk file header
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "mfree iwf\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "\n");

    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d float\n", N_particles);
    for (unsigned int i = 0; i < N_particles; i++)
    {
        fprintf(fp, "%f %f %f\n", particles->pos[i].x,
                particles->pos[i].y, particles->pos[i].z);
    }
    fprintf(fp, "\n");

    fprintf(fp, "CELLS %d %d\n", N_particles, 2 * N_particles);
    for (unsigned int i = 0; i < N_particles; i++)
    {

        fprintf(fp, "%d %d\n", 1, i);
    }
    fprintf(fp, "\n");

    fprintf(fp, "CELL_TYPES %d\n", N_particles);
    for (unsigned int i = 0; i < N_particles; i++)
    {

        fprintf(fp, "%d\n", 1);
    }
    fprintf(fp, "\n");

    fprintf(fp, "POINT_DATA %d\n", N_particles);

    /********************************************/
    // Current particle temperature
    fprintf(fp, "SCALARS Temperature float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (unsigned int i = 0; i < N_particles; i++)
    {
        fprintf(fp, "%f\n", particles->T[i]);
    }
    fprintf(fp, "\n");
    /********************************************/

    /********************************************/
    // Current particle hasing value
    fprintf(fp, "SCALARS hashing int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (unsigned int i = 0; i < N_particles; i++)
    {
        fprintf(fp, "%d\n", particles->hash[i]);
    }
    fprintf(fp, "\n");
    /********************************************/

    fclose(fp);
}

void vtk_writer_write_box(Grid* grid)
{
	char buf[256];
	sprintf(buf, "../results/grid.vtk");
	FILE *fp = fopen(buf, "w+");

	float3_t minCorner, maxCorner;
    minCorner = grid->get_minCorner();
    maxCorner = grid->get_maxCorner();

	const int num_face = 6;
	const int num_corner = 8;

	// generate 8 corners of tool bbox
	float3_t c000(minCorner.x, minCorner.y, minCorner.z);
	float3_t c100(maxCorner.x, minCorner.y, minCorner.z);
	float3_t c110(maxCorner.x, maxCorner.y, minCorner.z);
	float3_t c010(minCorner.x, maxCorner.y, minCorner.z);

	float3_t c001(minCorner.x, minCorner.y, maxCorner.z);
	float3_t c101(maxCorner.x, minCorner.y, maxCorner.z);
	float3_t c111(maxCorner.x, maxCorner.y, maxCorner.z);
	float3_t c011(minCorner.x, maxCorner.y, maxCorner.z);

	std::vector<float3_t> corners({c000, c100, c110, c010, c001, c101, c111, c011});

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");

	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n"); // Particle positions
	fprintf(fp, "POINTS %d float\n", num_corner);

	for (auto &it : corners)
	{
		fprintf(fp, "%f %f %f\n", it.x, it.y, it.z);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_face);
	for (int i = 0; i < num_face; i++)
	{
		fprintf(fp, "%d\n", 9);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_face, 5 * num_face);
	fprintf(fp, "4 %d %d %d %d\n", 0, 1, 2, 3);
	fprintf(fp, "4 %d %d %d %d\n", 0, 1, 5, 4);
	fprintf(fp, "4 %d %d %d %d\n", 1, 2, 6, 5);
	fprintf(fp, "4 %d %d %d %d\n", 0, 3, 7, 4);
	fprintf(fp, "4 %d %d %d %d\n", 3, 2, 6, 7);
	fprintf(fp, "4 %d %d %d %d\n", 4, 5, 6, 7);

	fprintf(fp, "\n");

	fclose(fp);
}