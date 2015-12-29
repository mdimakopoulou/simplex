#include <iostream>
#include <cmath>
#include <sstream>

#include "simplex_cpu.h"

#define EPSILON     1e-12

using namespace std;


void SimplexCPUSolver::printStats() const {
    cout << "Execution Statistics:\n  Iterations: " << steps <<
        "\n  Aborted: " << (aborted?"true":"false") <<
        "\n  No Solution: " << (noSolution?"true":"false") <<
        endl;
}

string SimplexCPUSolver::report() const {
    stringstream ss;
    ss << steps << "," << (aborted?1:0) << "," << (noSolution?1:0);
    return ss.str();
}

bool SimplexCPUSolver::init(Matrix& problem) {
    twoPhase = false;

    if (verbose) {
        cout << "CPU-Solver:: initializing\n";
    }

    determinePhaseCount(problem);

    if (twoPhase) {
        buildFirstPhaseTableau(problem);
    } else {
        buildSinglePhaseTableau(problem);
    }
    if (verbose) {
        cout << "CPU-Solver:: tableau built: " << tableau->rows() << "x" << tableau->columns() << "\n";
    }
    if (verbose > 1) {
        tableau->print();
    }

    return true;
}

void SimplexCPUSolver::solve(Matrix& problem) {
    if (twoPhase) {
        solveTwoPhase(problem);
    } else {
        solveSinglePhase(problem);
    }
}

void SimplexCPUSolver::solveTwoPhase(Matrix& problem) {
    if (verbose) {
        cout << "CPU-Solver:: prepare Phase I\n";
    }

    prepareToSolvePhaseI(problem);
    if (verbose) {
        cout << "CPU-Solver:: solve Phase I\n";
    }
    solveCurrentTableau();

    if (aborted || noSolution) {
        if (verbose) {
            cout << "CPU-Solver:: Phase I: no solution\n";
        }
        return;
    }

    if (fabs(tableau->at(tableau->rows() - 1, tableau->columns() - 1)) > EPSILON) {
        noSolution = true;
        if (verbose) {
            cout << "CPU-Solver:: Phase I: UNBOUND\n";
        }
        return;
    }

    buildSecondPhaseTableau(problem);
    if (verbose) {
        cout << "CPU-Solver:: Phase II: tableau built: " << tableau->rows() << "x" << tableau->columns() << "\n";
    }
    if (verbose > 1) {
        tableau->print();
    }

    zeroOutBasicSolutionObjective(problem);

    solveCurrentTableau();

    if (verbose) {
        cout << "CPU-Solver:: compiling solution\n";
    }

    if (!noSolution) {
        setupSolution();
    }

    delete tableau;

    if (verbose) {
        cout << "CPU-Solver:: completed\n";
    }
}

void SimplexCPUSolver::solveSinglePhase(Matrix& problem) {
    if (verbose) {
        cout << "CPU-Solver:: begins\n";
    }

    solveCurrentTableau();

    if (verbose) {
        cout << "CPU-Solver:: compiling solution\n";
    }

    if (!noSolution) {
        setupSolution();
    }

    delete tableau;

    if (verbose) {
        cout << "CPU-Solver:: completed\n";
    }
}

void SimplexCPUSolver::solveCurrentTableau() {
    steps = 0;
    aborted = false;
    while (hasNegativesInLastRow()) {
        ++steps;
        if (steps > maxSteps) {
            aborted = true;
            break;
        }
        if (verbose) {
            cout << "CPU-Solver:: entering step: " << steps << endl;
        }

        uint32_t pivotColumn = selectPivotColumn();
        if (pivotColumn == (uint32_t)-1) {
            saveTableau();

            steps--;
            break;
        }
        if (verbose) {
            cout << "CPU-Solver:: pivot column: " << (int)pivotColumn << endl;
        }
        uint32_t pivotRow = selectPivotRow(pivotColumn);
        if (pivotRow == (uint32_t)-1) {
            saveTableau();

            steps--;
            noSolution = true;
            break;
        }
        if (verbose) {
            cout << "CPU-Solver:: pivot row: " << (int)pivotRow << endl;
        }
        clearPivotColumn(pivotRow, pivotColumn);
        saveTableau();

        if (verbose > 1) {
            cout << "pivot: " << pivotRow << "x" << pivotColumn << endl;
            tableau->print();
        }
    }
}

void SimplexCPUSolver::buildFirstPhaseTableau(Matrix& problem) {
    slacks = problem.rows() - 1;
    tableau = new Matrix(problem.rows(), problem.columns() + slacks);

    tableau->copy(problem, 0, 0, 0, 0, problem.rows() - 1, problem.columns() - 1);
    tableau->copy(problem, 0, problem.columns() + slacks - 1, 0, problem.columns() - 1, problem.rows() - 1, 1);

    tableau->setIdentityAt(slacks, 0, problem.columns() - 1);
    // tableau->at(tableau->rows() - 1, tableau->columns() - 1) = 0;

    for (int i = 0; i < slacks; ++i) {
        tableau->at(tableau->rows() - 1, problem.columns() - 1 + i) = 1;
    }

    for (int i = 0; i < ge_no; ++i) {
        tableau->multiplyRow(le_no + i, -1);
    }
}

void SimplexCPUSolver::buildSecondPhaseTableau(Matrix& problem) {
    int32_t slacks_p1 = slacks;
    Matrix* tableau_p1 = tableau;

    slacks = le_no + ge_no;
    tableau = new Matrix(problem.rows(), problem.columns() + slacks);

    tableau->copy(*tableau_p1, 0, 0, 0, 0, tableau->rows() - 1, problem.columns() - 1);
    tableau->copy(*tableau_p1, 0, problem.columns() + slacks - 1, 0, tableau->columns() + slacks_p1 - 1, tableau->rows() - 1, 1);

    tableau->setIdentityAt(slacks, 0, problem.columns() - 1);
    tableau->at(tableau->rows() - 1, tableau->columns() - 1) = 0;

    tableau->copy(problem, problem.rows() - 1, 0, problem.rows() - 1, 0, 1, problem.columns() - 1);

    if (!minimize) {
        for (uint32_t i = 0; i < problem.columns() - 1; ++i) {
            tableau->at(problem.rows() - 1, i) *= -1;
        }
    }

    for (int i = 0; i < ge_no; ++i) {
        tableau->multiplyRow(le_no + i, -1);
    }

    prepareNewSolution(problem.columns());

    delete tableau_p1;
}

void SimplexCPUSolver::prepareToSolvePhaseI(Matrix& problem) {
    for (int i = 0; i < slacks; ++i) {
        tableau->maddRow(i, problem.rows() - 1, -1); 
    }

    if (verbose) {
        cout << "CPU-Solver:: prepared phase I tableau\n";
    }
    if (verbose > 1) {
        tableau->print();
    }
}

void SimplexCPUSolver::zeroOutBasicSolutionObjective(Matrix& problem) {
    for (int i = 0; i < (int)problem.columns() - 1; ++i) {
        int row = -1;
        for (int r = 0; r < (int)problem.rows() - 1; ++r) {
            if (fabs(tableau->at(r, i) - 1) < EPSILON) {
                if (row == -1) {
                    row = r;
                } else {
                    row = -1;
                    break;
                }
            } else if (fabs(tableau->at(r, i)) > EPSILON) {
                row = -1;
                break;
            }
        }

        if (row >= 0) {
            tableau->maddRow(row, tableau->rows() - 1, -tableau->at(tableau->rows() - 1, i));
        }
    }

    if (verbose) {
        cout << "CPU-Solver:: prepared phase II tableau\n";
    }
    if (verbose > 1) {
        tableau->print();
    }
}

void SimplexCPUSolver::buildSinglePhaseTableau(Matrix& problem) {
    slacks = le_no + ge_no;
    tableau = new Matrix(problem.rows(), problem.columns() + slacks);

    tableau->copy(problem, 0, 0, 0, 0, problem.rows(), problem.columns() - 1);
    tableau->copy(problem, 0, problem.columns() + slacks - 1, 0, problem.columns() - 1, problem.rows(), 1);

    tableau->setIdentityAt(slacks, 0, problem.columns() - 1);
    tableau->at(tableau->rows() - 1, tableau->columns() - 1) = 0;

    if (!minimize) {
        for (uint32_t i = 0; i < problem.columns() - 1; ++i) {
            tableau->at(problem.rows() - 1, i) *= -1;
        }
    }

    for (int i = 0; i < ge_no; ++i) {
        tableau->multiplyRow(le_no + i, -1);
    }

    prepareNewSolution(problem.columns());
}

bool SimplexCPUSolver::isActiveColumn(uint32_t column, uint32_t& nonZeroRow) const {
    int nonZeroRows = 0;
    for (uint32_t i = 0; i < tableau->rows(); ++i) {
        if (fabs(tableau->at(i, column)) > EPSILON) {
            nonZeroRow = i;
            ++nonZeroRows;
        }
    }
    return nonZeroRows == 1;
}

void SimplexCPUSolver::setupSolution() {
    Matrix& solution = *lastSolution;

    for (uint32_t c = 0; c < solution.columns() - 1; ++c) {
        uint32_t row = 0;
        if (isActiveColumn(c, row)) {
            double v = tableau->at(row, c);
            double n = tableau->at(row, tableau->columns() - 1);
            solution.at(0, c) = n / v;
        } else {
            solution.at(0, c) = 0;
        }
    }

    solution.at(0, solution.columns() - 1) = tableau->at(tableau->rows() - 1, tableau->columns() - 1);
}

uint32_t SimplexCPUSolver::selectPivotColumn() const {
    uint32_t lastRowIdx = tableau->rows() - 1;
    uint32_t maxAbsValueColumnIndex = (uint32_t) -1;
    double maxAbsValue = 0;
    for (uint32_t i = 0; i < tableau->columns() - 1; ++i) {
        double v = tableau->at(lastRowIdx, i);
        if (v >= EPSILON) {
            continue;
        }
        double absValue = fabs(v);
        if (absValue > maxAbsValue) {
            maxAbsValue = absValue;
            maxAbsValueColumnIndex = i;
        }
    }

    return maxAbsValueColumnIndex;
}

uint32_t SimplexCPUSolver::selectPivotRow(uint32_t column) const {
    uint32_t lastColumnIdx = tableau->columns() - 1;
    uint32_t minRatioRowIndex = (uint32_t) -1;
    double minRatio = 0;

    for (uint32_t i = 0; i < tableau->rows(); ++i) {
        double v = tableau->at(i, column);
        if (v <= EPSILON) {
            continue;
        }

        double ratio = tableau->at(i, lastColumnIdx) / v;
        if (ratio > 0 && (minRatioRowIndex == (uint32_t)-1 || minRatio > ratio)) {
            minRatio = ratio;
            minRatioRowIndex = i;
        }
    }

    return minRatioRowIndex;
}

void SimplexCPUSolver::clearPivotColumn(uint32_t pivotRow, uint32_t pivotColumn) {
    const double pivot = tableau->at(pivotRow, pivotColumn);

    for (uint32_t r = 0; r < tableau->rows(); ++r) {
        if (r == pivotRow) {
            continue;
        }

        double v = tableau->at(r, pivotColumn);
        if (fabs(v) < EPSILON) {
            continue;
        }

        tableau->maddRow(pivotRow, r, -v / pivot);

        // fix numerical skews...
        tableau->at(r, pivotColumn) = 0;
    }
    tableau->multiplyRow(pivotRow, 1 / pivot);
}

bool SimplexCPUSolver::hasNegativesInLastRow() const {
    for (uint32_t c = 0; c < tableau->columns() - 1; ++c) {
        if (tableau->at(tableau->rows() - 1, c) < 0) {
            return true;
        }
    }

    return false;
}

void SimplexCPUSolver::saveTableau() const {
    if (!prefix) {
        return;
    }

    char buf[512];
    sprintf(buf, "%s.%u.csv", prefix, steps);

    if (verbose) {
        cout << "CPU-Solver:: save: " << buf << endl;
    }
    tableau->toCSV(buf);

}

