#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "docs-mpi.h"
#include "state-mpi.h"

AlgorithmState* classify_docs(const Dataset *data);

void print_assignment(const int *assignment, int D);

#endif /* ALGORITHM_H */