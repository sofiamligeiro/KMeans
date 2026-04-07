#ifndef PARSER_H
#define PARSER_H

#include "docs.h"

Dataset* load_dataset(const char *filename);

void free_dataset(Dataset *data);

void print_dataset(const Dataset *data);

#endif /* PARSER_H */