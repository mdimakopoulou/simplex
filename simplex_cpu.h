#ifndef _SIMPLEX_CPU_H
#define _SIMPLEX_CPU_H

#include "simplex.h"


class SimplexCPUSolver : public Solver {
public:
    virtual void printStats() const;
    virtual std::string report() const;

    virtual bool init(Matrix& problem);
    virtual void solve(Matrix& problem);

private:
    void solveTwoPhase(Matrix& problem);
    void solveSinglePhase(Matrix& problem);

    void buildFirstPhaseTableau(Matrix& problem);
    void buildSecondPhaseTableau(Matrix& problem);
    void buildSinglePhaseTableau(Matrix& problem);

    void prepareToSolvePhaseI(Matrix& problem);
    void zeroOutBasicSolutionObjective(Matrix& problem);
    void solveCurrentTableau();

    bool isActiveColumn(uint32_t column, uint32_t& nonZeroRow) const;
    void setupSolution();
    uint32_t selectPivotColumn() const;
    uint32_t selectPivotRow(uint32_t column) const;
    void clearPivotColumn(uint32_t pivotRow, uint32_t pivotColumn);
    bool hasNegativesInLastRow() const;
    void saveTableau() const;

    Matrix* tableau;
    int32_t steps;
    int32_t slacks;
};

#endif
