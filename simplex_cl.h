#ifndef _SIMPLEX_CL_H
#define _SIMPLEX_CL_H

#include "simplex.h"
#include "clutil.h"


class SimplexCLSolver : public Solver {
public:
    SimplexCLSolver();
    ~SimplexCLSolver();

    bool initOK() const;

    virtual void printStats() const;
    virtual std::string report() const;

    virtual bool init(Matrix& problem);
    virtual void solve(Matrix& problem);

    void setUseBinaryDiv(bool flag) {
        useBinaryDividers = flag;
    }

private:
    bool buildBuffers(Matrix& problem);
    bool runSetup(const Matrix& problem);
    bool runSelectPivotColumn(const Matrix& problem, const cl_mem& tableauBuf);
    bool runSelectPivotColumnBinaryDiv(const Matrix& problem, const cl_mem& tableauBuf);
    bool runSelectPivotColumnItr(const Matrix& problem, const cl_mem& tableauBuf);
    bool runSelectPivotRow(const Matrix& problem, const cl_mem& tableauBuf);
    bool runSelectPivotRowBinaryDiv(const Matrix& problem, const cl_mem& tableauBuf);
    bool runSelectPivotRowItr(const Matrix& problem, const cl_mem& tableauBuf);
    bool runClearWithPivot(const Matrix& problem, const cl_mem& read, const cl_mem& write);
    bool runExtractSolution(const Matrix& problem, const cl_mem& tableau, const cl_mem& solutionBuf);
    void saveBuffer(const cl_mem& buffer, int rows, int columns);
    void showBuffer(const cl_mem& buffer, int rows, int columns);
    void showIntBuffer(const cl_mem& buffer, int rows, int columns);

    cl_command_queue cmdQ;
    cl_program program;

    cl_kernel setupTableau;
    cl_kernel findMin;
    cl_kernel selectNegatives;
    cl_kernel selectMinNegative;
    cl_kernel selectMinRatio;
    cl_kernel findMinRatio;
    cl_kernel copyPivotColumnRatios;
    cl_kernel clearWithPivot;
    cl_kernel buildSolutionTable;
    cl_kernel extractSolution;
    cl_kernel fixInfinities;

    cl_mem problemBuf;
    cl_mem tableauBuf0;
    cl_mem tableauBuf1;
    cl_mem indexBuf;

    cl_event lastEvent;

    uint32_t pivotColumn;
    uint32_t pivotRow;

    int32_t steps;
    uint32_t programsRun;
    uint32_t writeOps;
    uint32_t readOps;
    size_t bytesWriten;
    size_t bytesRead;
    size_t bytesAlloc;

    bool useBinaryDividers;
};

#endif
