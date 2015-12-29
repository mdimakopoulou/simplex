#ifndef _SIMPLEX_COMBINED_CL_H
#define _SIMPLEX_COMBINED_CL_H

#include "simplex.h"
#include "clutil.h"


class CombinedSimplexCLSolver : public Solver {
public:
    CombinedSimplexCLSolver();
    ~CombinedSimplexCLSolver();

    bool initOK() const;

    virtual void printStats() const;
    virtual std::string report() const;

    virtual bool init(Matrix& problem);
    virtual void solve(Matrix& problem);

private:
    void solveTwoPhase(Matrix& problem);
    void solveSinglePhase(Matrix& problem);
    cl_mem* solveCurrentTableau(Matrix& problem);

    bool prepareToSolvePhaseI(Matrix& problem);
    bool readObjectiveSolutionValue(double& value, cl_mem readBuf);
    bool buildSecondPhaseTableau(Matrix& problem, cl_mem readBuf, cl_mem writeBuf);
    bool zeroOutBasicSolutionObjective(Matrix& problem, cl_mem readBuf, cl_mem writeBuf);

    bool buildBuffers(Matrix& problem);
    bool runSetup(const Matrix& problem);
    bool runSelectPivot(const Matrix& problem, const cl_mem& tableauBuf);
    bool runClearWithPivot(const Matrix& problem, const cl_mem& read, const cl_mem& write);
    bool runExtractSolution(const Matrix& problem, const cl_mem& tableau, const cl_mem& solutionBuf);
    void saveBuffer(const cl_mem& buffer, int rows, int columns);
    void showBuffer(const cl_mem& buffer, int rows, int columns);
    void showIntBuffer(const cl_mem& buffer, int rows, int columns);

    cl_command_queue cmdQ;
    cl_program program;

    cl_kernel setupTableauPhaseI;
    cl_kernel setupTableauPhaseII;
    cl_kernel preparePhaseI;
    cl_kernel detectBasic;
    cl_kernel zeroOutBasic;

    cl_kernel setupTableau;
    cl_kernel selectPivot;
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

    int32_t tableSize[2];
    int32_t partition[2];
    size_t scannersNo;

    int32_t steps;
    int32_t slacks;
    int32_t slacks_p1;
    int32_t slacks_p2;

    uint32_t programsRun;
    uint32_t writeOps;
    uint32_t readOps;
    size_t bytesWriten;
    size_t bytesRead;
    size_t bytesAlloc;
};

#endif
