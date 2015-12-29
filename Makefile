
EXE := simplex

CC := g++

CFLAGS := -Wall -pedantic -I. -std=c++11 -O3 -s
#CFLAGS := -Wall -pedantic -I. -std=c++11 -g -ggdb

LDFLAGS := -lOpenCL


OBJECTS := main.o matrix.o simplex.o simplex_cpu.o simplex_cl.o \
		   clutil.o simplex_mcl.o simplex_wg_mcl.o simplex_comb_cl.o

all: $(OBJECTS)
	$(CC) -o $(EXE) $(OBJECTS) $(LDFLAGS) $(LIBS)

-include $(OBJECTS:.o=.d)

%.o: %.cc Makefile
	$(CC) -c -o $@ $< $(CFLAGS)
	$(CC) -MM $(CFLAGS) $< > $*.d

.PHONY: clean

clean:
	rm -f *.o *.d core $(EXE)

