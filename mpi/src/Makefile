# --- Compiler & Flags ---
MPICC    := mpicc
CFLAGS   := -O2 -fopenmp \
			-Wall -Wextra -g -std=c11 \
            -MMD -MP
LDFLAGS  := -fopenmp
LDLIBS   := -lm

# --- Target & Sources ---
TARGET := docs
SRCS   := docs-mpi.c parser-mpi.c algorithm-mpi.c state-mpi.c

# --- Derived paths ---
OBJDIR := obj
OBJS   := $(SRCS:%.c=$(OBJDIR)/%.o)
DEPS   := $(OBJS:.o=.d)

# --- Test directories ---
TESTS_IN_DIR       := tests/in
TESTS_EXPECTED_DIR := tests/expected
TESTS_OUT_DIR      := tests/out
TESTS_DIFF_DIR     := tests/diff
TESTS_TIME_DIR     := tests/time

# --- Runtime config ---
MPI_NP ?= 4   # override with: make run MPI_NP=8

# --- Rules ---
.PHONY: all clean run test benchmark

all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(MPICC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Compile
$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(MPICC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

-include $(DEPS)

run: $(TARGET)
	@echo "Looking for input files in $(TESTS_IN_DIR)..."
	@count=0; \
	FILES=$$(ls -v $(TESTS_IN_DIR)/* 2>/dev/null); \
	for f in $$FILES; do \
		[ -e "$$f" ] || continue; \
		count=$$((count+1)); \
		printf "%d: %s\n" "$$count" "$$(basename "$$f")"; \
	done; \
	[ "$$count" -gt 0 ] || { echo "No test files found."; exit 1; }; \
	printf "\nSelect a file by number: "; read choice; \
	current=0; \
	for f in $$FILES; do \
		[ -e "$$f" ] || continue; \
		current=$$((current+1)); \
		[ "$$current" -eq "$$choice" ] || continue; \
		echo "Running: mpirun -np $(MPI_NP) ./$(TARGET) $$f"; \
		mpirun -np $(MPI_NP) ./$(TARGET) "$$f"; break; \
	done

test: $(TARGET)
	@mkdir -p $(TESTS_OUT_DIR) $(TESTS_DIFF_DIR)
	@echo "Running MPI tests (NP=$(MPI_NP))..."; echo "========================================"
	@fails=0; passed=0; \
	FILES=$$(ls -v $(TESTS_IN_DIR)/*.in 2>/dev/null); \
	for infile in $$FILES; do \
		[ -e "$$infile" ] || continue; \
		base=$$(basename "$$infile" .in); \
		expected="$(TESTS_EXPECTED_DIR)/$$base.txt"; \
		out="$(TESTS_OUT_DIR)/$$base.out"; \
		err="$(TESTS_OUT_DIR)/$$base.err"; \
		diff_f="$(TESTS_DIFF_DIR)/$$base.diff"; \
		[ -f "$$expected" ] || { echo "[SKIP] $$base"; continue; }; \
		mpirun -np $(MPI_NP) ./$(TARGET) "$$infile" > "$$out" 2> "$$err"; \
		if diff -u "$$expected" "$$out" > "$$diff_f"; then \
			echo "✅ [PASS] $$base"; passed=$$((passed+1)); \
		else \
			echo "❌ [FAIL] $$base"; fails=$$((fails+1)); \
		fi; \
	done; \
	echo "========================================"; \
	[ "$$fails" -eq 0 ] && echo "All $$passed test(s) passed." || { echo "$$fails failed, $$passed passed."; exit 1; }

# TODO: run with "sbatch srun_docs.sh" or "mpirun -hostfile my-nodes.txt docs"
# TODO: benchmark for multiple different number of processes

clean:
	rm -rf $(OBJDIR) $(TARGET) $(MPI_TARGET)