# --- Variables ---
CC                 := gcc

CFLAGS             := -O2 -fopenmp -Wall -Wextra -g -std=c11
LDFLAGS            := -fopenmp
TARGET             := docs

SRCS               := docs-omp.c parser-omp.c algorithm-omp.c
OBJDIR             := obj
OBJS               := $(SRCS:%.c=$(OBJDIR)/%.o)

TESTS_IN_DIR       := ./tests/in
TESTS_EXPECTED_DIR := ./tests/expected
TESTS_OUT_DIR      := ./tests/out
TESTS_DIFF_DIR     := ./tests/diff
TESTS_TIME_DIR     := ./tests/time

# --- Rules ---
.PHONY: all clean run test benchmark

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS)

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

run: $(TARGET)
	@echo "Looking for input files in $(TESTS_IN_DIR)..."
	@count=0; \
	FILES=$$(ls -v $(TESTS_IN_DIR)/* 2>/dev/null || ls $(TESTS_IN_DIR)/*); \
	for f in $$FILES; do \
		[ -e "$$f" ] || continue; \
		count=$$((count+1)); \
		printf "%d: %s\n" "$$count" "$$(basename "$$f")"; \
	done; \
	if [ "$$count" -eq 0 ]; then \
		echo "Error: No test files found in $(TESTS_IN_DIR)"; \
		exit 1; \
	fi; \
	echo ""; \
	printf "Select a file by number: "; \
	read choice; \
	if ! [ "$$choice" -eq "$$choice" ] 2>/dev/null; then \
		echo "Error: Invalid input. Must be a number."; \
		exit 1; \
	fi; \
	if [ "$$choice" -lt 1 ] || [ "$$choice" -gt "$$count" ]; then \
		echo "Error: Number out of range."; \
		exit 1; \
	fi; \
	current=0; \
	for f in $$FILES; do \
		[ -e "$$f" ] || continue; \
		current=$$((current+1)); \
		if [ "$$current" -eq "$$choice" ]; then \
			echo "========================================"; \
			echo "Running: ./$(TARGET) $$f"; \
			echo "========================================"; \
			./$(TARGET) "$$f"; \
			break; \
		fi; \
	done

test: $(TARGET)
	@echo "Ensuring output directories exist..."
	@mkdir -p $(TESTS_OUT_DIR) $(TESTS_DIFF_DIR)
	@echo "Running tests..."
	@echo "========================================"
	@fails=0; \
	passed=0; \
	FILES=$$(ls -v $(TESTS_IN_DIR)/*.in 2>/dev/null || ls $(TESTS_IN_DIR)/*.in); \
	for infile in $$FILES; do \
		[ -e "$$infile" ] || continue; \
		base=$$(basename "$$infile" .in); \
		expected="$(TESTS_EXPECTED_DIR)/$$base.txt"; \
		out="$(TESTS_OUT_DIR)/$$base.out"; \
		err="$(TESTS_OUT_DIR)/$$base.err"; \
		difffile="$(TESTS_DIFF_DIR)/$$base.diff"; \
		if [ ! -f "$$expected" ]; then \
			echo "[SKIP] $$base (Expected file $$expected not found)"; \
			continue; \
		fi; \
		./$(TARGET) "$$infile" > "$$out" 2> "$$err"; \
		if diff -u "$$expected" "$$out" > "$$difffile"; then \
			echo "✅ [PASS] $$base"; \
			passed=$$((passed + 1)); \
		else \
			echo "❌ [FAIL] $$base -> Diff saved to $$difffile"; \
			fails=$$((fails + 1)); \
		fi; \
	done; \
	echo "========================================"; \
	if [ "$$fails" -gt 0 ]; then \
		echo "Result: $$fails test(s) failed, $$passed passed."; \
		exit 1; \
	else \
		echo "Result: All $$passed test(s) passed successfully!"; \
		exit 0; \
	fi

benchmark: $(TARGET)
	@mkdir -p $(TESTS_TIME_DIR)
	@echo "Looking for input files in $(TESTS_IN_DIR)..."
	@count=0; \
	FILES=$$(ls -v $(TESTS_IN_DIR)/* 2>/dev/null || ls $(TESTS_IN_DIR)/*); \
	for f in $$FILES; do \
	   [ -e "$$f" ] || continue; \
	   count=$$((count+1)); \
	   printf "%d: %s\n" "$$count" "$$(basename "$$f")"; \
	done; \
	if [ "$$count" -eq 0 ]; then \
	   echo "Error: No test files found in $(TESTS_IN_DIR)"; \
	   exit 1; \
	fi; \
	echo ""; \
	printf "Select a file by number to benchmark: "; \
	read choice; \
	if ! [ "$$choice" -eq "$$choice" ] 2>/dev/null; then \
	   echo "Error: Invalid input. Must be a number."; \
	   exit 1; \
	fi; \
	if [ "$$choice" -lt 1 ] || [ "$$choice" -gt "$$count" ]; then \
	   echo "Error: Number out of range."; \
	   exit 1; \
	fi; \
	current=0; \
	for f in $$FILES; do \
	   [ -e "$$f" ] || continue; \
	   current=$$((current+1)); \
	   if [ "$$current" -eq "$$choice" ]; then \
	      target_file="$$f"; \
	      break; \
	   fi; \
	done; \
	printf "How many runs per thread count? (Suggested: 3 for >50s, 5 for 10-50s, 10 for <10s): "; \
	read runs; \
	if ! [ "$$runs" -eq "$$runs" ] 2>/dev/null || [ "$$runs" -lt 1 ]; then \
	   echo "Invalid input. Defaulting to 5 runs."; \
	   runs=5; \
	fi; \
	base_name=$$(basename "$$target_file"); \
	report_file="$(TESTS_TIME_DIR)/$${base_name%.*}_time.txt"; \
	echo "========================================"; \
	echo "Benchmarking: $$base_name ($$runs runs per configuration)"; \
	echo "========================================"; \
	echo "Test: $$base_name" > "$$report_file"; \
	echo "Runs per thread count: $$runs" >> "$$report_file"; \
	echo "========================================" >> "$$report_file"; \
	for t in 1 2 4 6 8 12 16; do \
	   echo "Testing with $$t threads..."; \
	   echo "Threads: $$t" >> "$$report_file"; \
	   echo "-------------------" >> "$$report_file"; \
	   total=0; \
	   for i in $$(seq 1 $$runs); do \
	      printf "  Run $$i/$$runs... "; \
	      OMP_NUM_THREADS=$$t ./$(TARGET) "$$target_file" > /dev/null 2> .tmp_stderr; \
	      val=$$(grep -oE '[0-9]+\.[0-9]+s' .tmp_stderr | tr -d 's'); \
	      echo "  Run $$i: $${val}s" >> "$$report_file"; \
	      echo "$${val}s"; \
	      total=$$(awk "BEGIN {print $$total + $$val}"); \
	   done; \
	   avg=$$(awk "BEGIN {printf \"%.2f\", $$total / $$runs}"); \
	   echo "-------------------" >> "$$report_file"; \
	   echo "Average: $${avg}s" >> "$$report_file"; \
	   echo "========================================" >> "$$report_file"; \
	   echo "Average for $$t threads: $${avg}s"; \
	   echo ""; \
	done; \
	echo "Done! Clean results saved to: $$report_file"; \
	rm -f .tmp_stderr

clean:
	rm -rf $(OBJDIR) $(TARGET)
