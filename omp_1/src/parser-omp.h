#ifndef PARSER_OMP_H
#define PARSER_OMP_H

#include "docs-omp.h"

#define NUMBER_EXPECTED_HEADER_ITEMS 3
#define NUMBER_EXPECTED_SCALAR_ITEMS 1

Dataset* load_dataset(const char *filename);

void free_dataset(Dataset *data);

void print_dataset(const Dataset *data);

#endif /* PARSER_OMP_H */