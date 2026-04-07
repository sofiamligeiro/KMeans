# Parallel Document Classification
This project was developed as a group effort by three students for the Parallel and Distributed Computing(25/26) course.
The goal was to implement and optimize a document classification algorithm (based on the K-Means clustering method) to assign documents 
to cabinets by maximizing subject similarity across different parallel architectures

## Project Structure

The repository is organized to reflect the evolutionary development and optimization of the algorithm:

* **`Assignment.pdf`** The official project description containing the problem context and requirements.
* **`serial/`** The initial **baseline** implementation. A strictly sequential C version used as the reference point for all performance benchmarks.
* **`serial_simd/`** An enhanced version of the serial code utilizing **SIMD** instructions to leverage CPU-level vector parallelism.
* **`omp_1/`** The first **OpenMP** implementation. This folder includes the first performance study report and scalability comparisons against the baseline.
* **`omp_2/`** An improved **OpenMP** version of the previous one.
* **`mpi/`** The **MPI** implementation designed for **Distributed-Memory** systems. This version was tested on a cluster (Deucalion supercomputer) and contains the final performance report.

## Authors
* **Artur Krystopchuk - arturkrystopchuk@tecnico.ulisboa.pt**
* **João Martins - joao.bernardo.mota.martins@tecnico.ulisboa.pt**
* **Sofia Ligeiro - sofiamligeiro@tecnico.ulisboa.pt**


