/**
 * Parallel and Distributed Computing 25/26
 * 
 * GROUP 33:
 * Artur Krystopchuk (104145) - arturkrystopchuk@tecnico.ulisboa.pt
 * João Martins (106819) - joao.bernardo.mota.martins@tecnico.ulisboa.pt
 * Sofia Ligeiro (116046) - sofiamligeiro@tecnico.ulisboa.pt
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "parser-omp.h"
#include "docs-omp.h"
#include "algorithm-omp.h"

int main(int argc, char *argv[])  {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }
    const char *filename = argv[1];
    Dataset *data = load_dataset(filename);
    int* assignment;
    double exec_time;

    if (data != NULL) {
        exec_time = -omp_get_wtime();
        assignment = classify_docs(data);
        exec_time += omp_get_wtime();

        fprintf(stderr, "%.1fs\n", exec_time);
        print_assigment(assignment, data->num_documents);

        free_dataset(data);
        free(assignment);
    }

    return 0;
}