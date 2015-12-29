#ifndef _SIMPLEX_H
#define _SIMPLEX_H

#include <cstddef>
#include <exception>
#include <string>

#include "matrix.h"
#include "main.h"


class Solver {
public:
    Solver()
        : lastSolution(nullptr), maxSteps(INT32_MAX), aborted(false), noSolution(false), prefix(nullptr),
          minimize(true), le_no(0), ge_no(0), eq_no(0), forcePhaseI(false)
    {
    }
    virtual ~Solver() {
        if (lastSolution) delete lastSolution;
    }

    virtual void printStats() const = 0;
    virtual std::string report() const = 0;

    virtual bool init(Matrix& problem) = 0;
    virtual void solve(Matrix& problem) = 0;

    void setMaxIterations(int32_t maxitr) { maxSteps = maxitr; }

    void setTableauSavePrefix(char* prefix) { this->prefix = prefix; }

    Matrix* getLastSolution() const {
        return lastSolution;
    }

    void setMinimize(bool flag) { minimize = flag; }
    void setConstraintTypeCounts(int le, int ge, int eq) {
        le_no = le;
        ge_no = ge;
        eq_no = eq;
    }

    void setForcePhaseI(bool flag) {
        forcePhaseI = flag;
    }

protected:
    Matrix* lastSolution;
    int32_t maxSteps;
    bool aborted;
    bool noSolution;
    char* prefix;

    bool minimize;
    int le_no;
    int ge_no;
    int eq_no;

    bool forcePhaseI;
    bool twoPhase;

    void prepareNewSolution(uint32_t size) {
        if (lastSolution) {
            delete lastSolution;
        }

        lastSolution = new Matrix(1, size);
    }

    void determinePhaseCount(Matrix& problem);
};

#endif
