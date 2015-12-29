#ifndef _SIMPLEX_MONOLITHIC_CL_H
#define _SIMPLEX_MONOLITHIC_CL_H

#include "simplex.h"
#include "clutil.h"


class MonolithicSimplexCLSolver : public Solver {
public:
    MonolithicSimplexCLSolver();
    ~MonolithicSimplexCLSolver();

    bool initOK() const;

    virtual void printStats() const;
    virtual std::string report() const;

    virtual bool init(Matrix& problem);
    virtual void solve(Matrix& problem);

private:
    bool buildBuffers(Matrix& problem);
    bool runSetup(const Matrix& problem);
    bool runSimplex(const Matrix& problem);
    bool extractSolution(const Matrix& problem);
    void showBuffer(const cl_mem& buffer, int rows, int columns);

    cl_command_queue cmdQ;
    cl_program program;

    cl_kernel setupTableau;
    cl_kernel simplex;

    cl_mem problemBuf;
    cl_mem tableauBuf0;
    cl_mem tableauBuf1;
    cl_mem indexBuf;

    cl_event lastEvent;

    uint32_t rowPartition;
    uint32_t colPartition;

    uint32_t steps;
    size_t bytesWriten;
    size_t bytesRead;
    size_t bytesAlloc;
};

#endif
