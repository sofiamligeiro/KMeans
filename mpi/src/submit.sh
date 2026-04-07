#!/bin/bash
#SBATCH --account=f202500009hpcvlabistul2x
#SBATCH --partition=normal-x86


#=============CHANGE THESE PARAMETERS =================
#SBATCH --time=00:10:00              
#SBATCH --nodes=2
#SBATCH --ntasks=256               
#SBATCH --cpus-per-task=1
#SBATCH --output=mpi_test_256.out      


module load OpenMPI

INPUT_FILE="tests/in/ex200-400000-10-.1.in"

echo "--- STARTING TESTS WITH $SLURM_NTASKS PROCESSES ---"

for i in 1 2 3
do
   echo ">>> RUN $i"
   time mpirun ./docs "$INPUT_FILE"
   echo "---------------------------------------"
done