#ifndef PARSER_MPI_H
#define PARSER_MPI_H

#include "docs-mpi.h"

#define NUMBER_EXPECTED_HEADER_ITEMS 3
#define NUMBER_EXPECTED_SCALAR_ITEMS 1

Dataset *load_dataset(const char *filename, int rank, int size);

void compute_local_range(int D, int rank, int size,
                         int *local_start, int *local_num_documents);

void free_dataset(Dataset *data);

void print_dataset(const Dataset *data, int rank);

#endif /* PARSER_MPI_H */