#include <iostream>
#include "simplex.h"

using namespace std;


void Solver::determinePhaseCount(Matrix& problem) {
    if (forcePhaseI) {
        twoPhase = true;
        return;
    }

    bool has_negatives = false;
    for (uint32_t i = 0; i < problem.columns() - 1; ++i) {
        if (problem.at(problem.rows() - 1, i) < 0) {
            has_negatives = true;
        }
    }

    twoPhase = !has_negatives;
    if (verbose) {
        cout << "XX-Solver:: phase count: " << (twoPhase ? 2 : 1) << "\n";
    }
}

