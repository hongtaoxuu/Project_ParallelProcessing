BINARY_NAME = dgemm_x86
CC			= gcc
CFLAGS		= -O0 -march=skylake-avx512 -w -lpthread -fopenmp
MKLPATH		= /share/app/intel2025/mkl/2025.0
LDFLAGS		= -L$(MKLPATH)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64
INCFLAGS	= -I$(MKLPATH)/include


SRC			= $(wildcard *.c)
build : $(BINARY_NAME)

$(BINARY_NAME): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(SRC) -o $(BINARY_NAME)

clean:
	rm $(BINARY_NAME)