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

#include <iostream>
#include <vector>
#include <memory>
#include "particle.h"
#include "type.h"
#include <vtk_writer.h>
#include <interaction.h>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>
#include <grid.h>
#include <thread>
#include <barrier>

void update(Particles *particles, float_t dt, size_t start, size_t end)
{
    for (size_t i = start; i < end; i++)
    {
        // update temperature of particles
        particles->T[i] += particles->T_t[i] * dt;
    }
}

/**
 * @brief Initializes the particles with the given parameters.
 *
 * This function creates particles based on the specified dimensions and properties.
 * The particles are stored in the provided `Particles` object.
 *
 * @param particles A pointer to the `Particles` object where the particles will be stored.
 * @param nx The number of particles in the x-direction.
 * @param ny The number of particles in the y-direction.
 * @param nz The number of particles in the z-direction.
 * @param dx The spacing between particles in the x-direction.
 * @param dy The spacing between particles in the y-direction.
 * @param dz The spacing between particles in the z-direction.
 * @param hf The smoothing length factor.
 */
void initialize_particles(Particles *particles, int nx, int ny, int nz,
                          float_t dx, float_t dy, float_t dz, float_t hf)
{
    float_t thermal_cp = 850.;           // units: J/kgK 880
    float_t thermal_k = 157.;            // units: W/mK 155
    float_t rho = 2830.;                 // units: kg/m^3 2800
    float_t mass = 2830. * dx * dy * dz; // units: kg
    float_t h = hf * dx;                 // units: m
    float_t T = 20.;                     // units: C (initial temperature)
    float_t T_boundary = 500.;           // units: C  (boundary temperature)

    // particles containers
    std::vector<float3_t> pos_vec;
    std::vector<float_t> T_vec;
    std::vector<float_t> cp_vec;
    std::vector<float_t> k_vec;
    std::vector<float_t> rho_vec;
    std::vector<float_t> h_vec;
    std::vector<float_t> mass_vec;

    // create particles
    for (size_t i = 0; i < nx; i++)
    {
        for (size_t j = 0; j < ny; j++)
        {
            for (size_t k = 0; k < nz; k++)
            {
                pos_vec.push_back(make_float3_t(i * dx, j * dx, k * dx)); // set position of particles
                cp_vec.push_back(thermal_cp);
                k_vec.push_back(thermal_k);
                rho_vec.push_back(rho);
                mass_vec.push_back(mass);
                h_vec.push_back(h);
                T_vec.push_back(i == 0 ? T_boundary : T); // set boundary temperature and initial temperature
            }
        }
    }

    particles->set_properties(pos_vec, T_vec, cp_vec, k_vec, rho_vec, h_vec, mass_vec); // set properties of particles
}

/**
 * Runs the simulation for a given set of particles and grid.
 *
 * @param particles A pointer to the Particles object.
 * @param grid A pointer to the Grid object.
 * @param total_time The total time for the simulation.
 * @param dt The time step for the simulation.
 * @param start The starting index of the particles for this thread.
 * @param end The ending index of the particles for this thread.
 * @param bar A reference to the std::barrier object for synchronization.
 * @param thread_id The ID of the current thread.
 */
void run_simulation(Particles *particles, Grid *grid, float_t total_time, float_t dt, size_t start, size_t end, std::barrier<> &bar, size_t thread_id)
{
    // for tracking the progress of the simulation
    float_t percent_increment = 0.01;
    float_t next_percent = percent_increment;
    unsigned int step = 0;

    // write initial state
    if (thread_id == 0)
    {
        vtk_writer_write(particles, step);
    }
    // wait for thread 0 to finish writing the initial state
    bar.arrive_and_wait();

    for (float_t t = 0; t < total_time; t += dt)
    {
        // calculate heat transfer between particles T_t
        interactions_heat(particles, grid, start, end);

        // wait for all threads to finish calculating heat transfer
        bar.arrive_and_wait();

        // update temperature T of particles
        update(particles, dt, start, end);

        // wait for all threads to finish updating temperature
        bar.arrive_and_wait();

        // write vtk file every 1% of the total time using thread 0
        if (thread_id == 0)
        {
            float_t current_percent = t / total_time;

            if (current_percent >= next_percent)
            {
                std::cout << "Completion: " << int(current_percent * 100) << "%" << std::endl;
                next_percent += percent_increment;
                // write vtk file
                vtk_writer_write(particles, step);
            }
        }

        // wait for thread 0 to finish writing vtk file
        bar.arrive_and_wait();

        step++;
    }
}

int main(int argc, char const *argv[])
{
    // make results directoy if not present
    struct stat st = {0};
    if (stat("../results", &st) == -1)
    {
        mkdir("../results", 0777);
    }
    // clear files from result directory
    int ret;
    ret = system("rm ../results/*.vtk");

    // cube size
    float_t lx = 0.1;
    float_t ly = 0.1;
    float_t lz = 0.1;
    // Space between particles
    float_t dx = 0.005;
    float_t dy = dx;
    float_t dz = dx;
    // number of particles in each direction
    int nx = lx / dx;
    int ny = ly / dy;
    int nz = lz / dz;
    // total number of particles
    int N = nx * ny * nz;
    // smoothing length factor
    float_t hf = 1.2;

    Particles *particles = new Particles(N);
    initialize_particles(particles, nx, ny, nz, dx, dy, dz, hf);

    // create grid with min and max corners and cell size equal to 2*hf*dx
    // why 2*hf*dx? Because the kernel function is defined for 2*hf*dx
    // see the following line float_t fourh2 = 4 * hi * hi; in kernel.h file
    Grid *grid = new Grid(particles->pos[0], particles->pos[N - 1], 2 * hf * dx);
    vtk_writer_write_box(grid);
    grid->compute_hashes(particles);
    grid->hash_sort(particles);
    grid->compute_cellStartEnd(particles);
    vtk_writer_write(particles, 0);

    float_t total_time = 10; // units: seconds (simulation time)
    float_t dt = 0.01;       // units: seconds (time step)
    // Start time for tracking how long it takes
    auto start_time = std::chrono::high_resolution_clock::now();

    // number of threads
    size_t num_threads = 10;
    // container for threads
    std::vector<std::thread> threads;
    // barrier for thread synchronization
    std::barrier bar(num_threads);
    // iterate over the number of threads
    for (size_t i = 0; i < num_threads; i++)
    {
        // total number of particles in the simulation
        size_t N = particles->N;
        // number of particles per thread
        size_t N_per_thread = N / num_threads;
        // start and end indices for the current thread
        size_t start = i * N_per_thread;
        size_t end = (i + 1) * N_per_thread;
        // if it is the last thread, assign the remaining particles to it
        if (i == num_threads - 1)
        {
            end = N;
        }
        // create a thread and run the simulation
        threads.emplace_back(run_simulation, particles, grid, total_time, dt, i * N / num_threads, (i + 1) * N / num_threads, std::ref(bar), i);
    }

    // join all threads
    for (auto &t : threads)
    {
        t.join();
    }

    // run_simulation(particles, grid, total_time, dt); // run simulation
    //  End time for tracking how long it takes
    auto end_time = std::chrono::high_resolution_clock::now();
    // calculate time used
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "Simulation time:" << total_time << " seconds" << std::endl;
    std::cout << "Time step:" << dt << " seconds" << std::endl;
    std::cout << "Time used: " << duration.count() << " seconds" << std::endl;

    delete particles; // freeing the memory
    return 0;
}