#include <iostream>
#include <cmath>
#include <sstream>
#include <chrono>
#include <ctime>
#include <cassert>

#include "clutil.h"
#include "simplex_comb_cl.h"

using namespace std;


#define E_CL(e, where) \
    if ((e) != CL_SUCCESS) { \
        cerr << "CL-Solver[" where "] ERROR(" << (-(e)) << "): " << clErrorString(e) << endl; \
        return false; \
    }

CombinedSimplexCLSolver::CombinedSimplexCLSolver() {
    problemBuf = 0;
    tableauBuf0 = 0;
    tableauBuf1 = 0;
    indexBuf = 0;

    CLSys& cl = CLSys::getInstance();

    cmdQ = cl.createQueue();
    program = cl.buildProgram(loadTextFile("cl/simplex_comb.cl"));
    setupTableauPhaseI = cl.createKernel(program, "setupTableauPhaseI");
    setupTableauPhaseII = cl.createKernel(program, "setupTableauPhaseII");
    preparePhaseI = cl.createKernel(program, "prepareToSolvePhaseI");
    detectBasic = cl.createKernel(program, "detectBasicSolutionMembers");
    zeroOutBasic = cl.createKernel(program, "zeroOutBasicSolutionMembersObjective");
    setupTableau = cl.createKernel(program, "setupTableau");
    selectPivot = cl.createKernel(program, "selectPivot");
    clearWithPivot = cl.createKernel(program, "clearWithPivot");
    buildSolutionTable = cl.createKernel(program, "buildSolutionTable");
    extractSolution = cl.createKernel(program, "extractSolution");
    fixInfinities = cl.createKernel(program, "fixInfinities");
}

CombinedSimplexCLSolver::~CombinedSimplexCLSolver() {
    if (fixInfinities) clReleaseKernel(fixInfinities);
    if (extractSolution) clReleaseKernel(extractSolution);
    if (buildSolutionTable) clReleaseKernel(buildSolutionTable);
    if (clearWithPivot) clReleaseKernel(clearWithPivot);
    if (selectPivot) clReleaseKernel(selectPivot);
    if (setupTableauPhaseI) clReleaseKernel(setupTableauPhaseI);
    if (setupTableauPhaseII) clReleaseKernel(setupTableauPhaseII);
    if (preparePhaseI) clReleaseKernel(preparePhaseI);
    if (setupTableau) clReleaseKernel(setupTableau);
    if (detectBasic) clReleaseKernel(detectBasic);
    if (zeroOutBasic) clReleaseKernel(zeroOutBasic);
    if (program) clReleaseProgram(program);
    if (problemBuf) clReleaseMemObject(problemBuf);
    if (tableauBuf0) clReleaseMemObject(tableauBuf0);
    if (tableauBuf1) clReleaseMemObject(tableauBuf1);
    if (indexBuf) clReleaseMemObject(indexBuf);
    if (cmdQ) clReleaseCommandQueue(cmdQ);
}

bool CombinedSimplexCLSolver::initOK() const {
    return cmdQ && program && setupTableau &&
           selectPivot && extractSolution && fixInfinities;
}

void CombinedSimplexCLSolver::printStats() const {
    cout << "Execution Statistics:\n  Iterations: " << steps <<
        "\n  Aborted: " << (aborted?"true":"false") <<
        "\n  No Solution: " << (noSolution?"true":"false") <<
        "\n  CL Programs invocations: " << programsRun <<
        "\n  CL Write ops: " << writeOps <<
        "\n  CL Read ops: " << readOps <<
        "\n  CL Allocation (bytes): " << bytesAlloc <<
        "\n  CL Writen (bytes): " << bytesWriten <<
        "\n  CL Read (bytes): " << bytesRead <<
        endl;
}

string CombinedSimplexCLSolver::report() const {
    stringstream ss;
    ss << steps << "," << (aborted?1:0) << "," << (noSolution?1:0);
    ss << ":";
    ss << programsRun << ",";
    ss << writeOps << ",";
    ss << readOps << ",";
    ss << bytesAlloc << ",";
    ss << bytesWriten << ",";
    ss << bytesRead;
    return ss.str();
}

bool CombinedSimplexCLSolver::init(Matrix& problem) {
    steps = programsRun = writeOps = readOps = 0;
    bytesWriten = bytesRead = bytesAlloc = 0;
    twoPhase = false;

    if (verbose) {
        cout << "CL-Solver:: init\n";
    }

    determinePhaseCount(problem);

    if (!buildBuffers(problem)) return false;
    if (!runSetup(problem)) return false;
    return true;
}

void CombinedSimplexCLSolver::solve(Matrix& problem) {
    if (twoPhase) {
        solveTwoPhase(problem);
    } else {
        solveSinglePhase(problem);
    }
}

void CombinedSimplexCLSolver::solveTwoPhase(Matrix& problem) {
    if (verbose) {
        cout << "CL-Solver:: prepare Phase I\n";
    }

    prepareToSolvePhaseI(problem);
    if (verbose) {
        cout << "CL-Solver:: solve Phase I\n";
    }

    cl_mem* doubleBuf = solveCurrentTableau(problem);

    if (aborted || noSolution) {
        if (verbose) {
            cout << "CL-Solver:: Phase I: no solution\n";
        }
        delete [] doubleBuf;
        return;
    }

    double v;
    if (!readObjectiveSolutionValue(v, doubleBuf[steps%2])) return;
    if (fabs(v) > 1e-10) {
        noSolution = true;
        if (verbose) {
            cout << "CL-Solver:: Phase I: UNBOUND\n";
        }
        delete [] doubleBuf;
        return;
    }

    if (!buildSecondPhaseTableau(problem, doubleBuf[steps%2], doubleBuf[(steps + 1)%2])) {
        delete [] doubleBuf;
        return;
    }

    if (!zeroOutBasicSolutionObjective(problem, doubleBuf[steps%2], doubleBuf[(steps + 1)%2])) {
        delete [] doubleBuf;
        return;
    }

    if (steps % 2 == 0) {
        cl_mem tmp = tableauBuf0;
        tableauBuf0 = tableauBuf1;
        tableauBuf1 = tmp;
    }

    delete [] doubleBuf;
    doubleBuf = solveCurrentTableau(problem);

    if (verbose) {
        cout << "CL-Solver:: compiling solution\n";
    }

    prepareNewSolution(problem.columns());
    if (!aborted && !noSolution) {
        assert(doubleBuf);
        if (!runExtractSolution(problem, doubleBuf[steps%2], doubleBuf[(steps + 1)%2])) {
            delete [] doubleBuf;
            return;
        }
        delete [] doubleBuf;
    }

    if (verbose) {
        cout << "CL-Solver:: completed\n";
    }
}

void CombinedSimplexCLSolver::solveSinglePhase(Matrix& problem) {
    if (verbose) {
        cout << "CL-Solver:: begins\n";
    }

    cl_mem* doubleBuf = solveCurrentTableau(problem);

    if (verbose) {
        cout << "CL-Solver:: compiling solution\n";
    }

    prepareNewSolution(problem.columns());
    if (!aborted && !noSolution) {
        assert(doubleBuf);
        if (!runExtractSolution(problem, doubleBuf[steps%2], doubleBuf[(steps + 1)%2])) {
            delete [] doubleBuf;
            return;
        }
        delete [] doubleBuf;
    }

    if (verbose) {
        cout << "CL-Solver:: completed\n";
    }
}

cl_mem* CombinedSimplexCLSolver::solveCurrentTableau(Matrix& problem) {
    cl_mem* doubleBuf = new cl_mem[2];
    doubleBuf[0] = tableauBuf0;
    doubleBuf[1] = tableauBuf1;
    noSolution = false;
    aborted = true;

    while(steps < maxSteps) {
        cl_mem read_buf = doubleBuf[steps%2];
        cl_mem write_buf = doubleBuf[(steps + 1)%2];
        ++steps;

        if (verbose) {
            cout << "CL-Solver:: entering step: " << steps << endl;
        }

        if (!runSelectPivot(problem, read_buf))  {
            delete [] doubleBuf;
            return nullptr;
        }
        if (pivotColumn == (uint32_t)-1) {
            saveBuffer(write_buf, tableSize[0], tableSize[1]);

            --steps;
            aborted = false;
            break; // no pivot? we found a solution :)
        }

        if (pivotRow == (uint32_t)-1) {
            saveBuffer(write_buf, tableSize[0], tableSize[1]);

            --steps;
            noSolution = true;
            aborted = false;
            break; // no pivot? we found a solution :)
        }

        if (!runClearWithPivot(problem, read_buf, write_buf)) {
            delete [] doubleBuf;
            return nullptr;
        }
        
        saveBuffer(write_buf, tableSize[0], tableSize[1]);

        if (verbose > 1) {
            showBuffer(write_buf, tableSize[0], tableSize[1]);
        }
    }

    return doubleBuf;
}

bool CombinedSimplexCLSolver::buildBuffers(Matrix& problem) {
    CLSys& cl = CLSys::getInstance();

    cl_int error = CL_SUCCESS;
    bytesAlloc += problem.size() * sizeof(double);
    writeOps++;
    bytesWriten += problem.size() * sizeof(double);
    problemBuf = clCreateBuffer(cl.ctx(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, problem.size() * sizeof(double), problem.rawptr(), &error);
    E_CL(error, "C(problem)");

    slacks_p1 = twoPhase ? problem.rows() - 1 : 0;
    slacks_p2 = ge_no + le_no;
    int32_t tableauColumns = std::max(slacks_p1, slacks_p2) + problem.columns();

    uint32_t tableauBufSize = problem.rows() * tableauColumns * sizeof(double);
    bytesAlloc += tableauBufSize;
    tableauBuf0 = clCreateBuffer(cl.ctx(), CL_MEM_READ_WRITE, tableauBufSize, 0, &error);
    E_CL(error, "C(tableau-0)");

    bytesAlloc += tableauBufSize;
    tableauBuf1 = clCreateBuffer(cl.ctx(), CL_MEM_READ_WRITE, tableauBufSize, 0, &error);
    E_CL(error, "C(tableau-1)");

    tableSize[0] = problem.rows();
    tableSize[1] = tableauColumns;

    partition[0] = std::max(1, (int32_t)std::min((int32_t)CLSys::getInstance().deviceWorkGroupSize(), (int32_t)ceil(sqrt(tableSize[1]) - 1)));
    partition[1] = std::max(1, (int32_t)std::min((int32_t)CLSys::getInstance().deviceWorkGroupSize(), (int32_t)ceil(sqrt(tableSize[0] - 1))));
    scannersNo = std::max(partition[0], partition[1]);
    uint32_t indexBufSz = std::max(std::max(2, tableSize[1] - 1), (int32_t)scannersNo + 1);

    bytesAlloc += sizeof(uint32_t) * indexBufSz;
    indexBuf = clCreateBuffer(cl.ctx(), CL_MEM_READ_WRITE, sizeof(uint32_t) * indexBufSz, 0, &error);
    E_CL(error, "C(index)");

    return true;
}

bool CombinedSimplexCLSolver::runSetup(const Matrix& problem) {
    cl_int error = CL_SUCCESS;

    const uint32_t variables = problem.columns() - 1;
    const size_t setupWorkSize[] = { (size_t)tableSize[0], (size_t)tableSize[1] };
    cl_event setupComplete;
    ++programsRun;
    if (!twoPhase) {
        const int32_t objective_sign = minimize ? 1 : -1;
        clSetKernelArg(setupTableau, 0, sizeof(cl_mem), &problemBuf);
        clSetKernelArg(setupTableau, 1, sizeof(cl_mem), &tableauBuf0);
        clSetKernelArg(setupTableau, 2, sizeof(uint32_t), &variables);
        clSetKernelArg(setupTableau, 3, sizeof(int32_t), &objective_sign);

        error = clEnqueueNDRangeKernel(cmdQ, setupTableau, 2, nullptr, setupWorkSize, nullptr, 0, nullptr, &setupComplete);
    } else {
        clSetKernelArg(setupTableauPhaseI, 0, sizeof(cl_mem), &problemBuf);
        clSetKernelArg(setupTableauPhaseI, 1, sizeof(cl_mem), &tableauBuf0);
        clSetKernelArg(setupTableauPhaseI, 2, sizeof(uint32_t), &variables);

        error = clEnqueueNDRangeKernel(cmdQ, setupTableauPhaseI, 2, nullptr, setupWorkSize, nullptr, 0, nullptr, &setupComplete);
    }
    E_CL(error, "Q(setup)");
    lastEvent = setupComplete;

    if (verbose > 1) {
        showBuffer(tableauBuf0, tableSize[0], tableSize[1]);
    }

    return true;
}

bool CombinedSimplexCLSolver::prepareToSolvePhaseI(Matrix& problem) {
    cl_int error = CL_SUCCESS;
    cl_event complete;
    ++programsRun;

    clSetKernelArg(preparePhaseI, 0, sizeof(cl_mem), &tableauBuf0);
    clSetKernelArg(preparePhaseI, 1, sizeof(uint32_t) * 2, tableSize);
    clSetKernelArg(preparePhaseI, 2, sizeof(int32_t), &slacks_p1);

    const size_t count = problem.columns() + slacks_p1 + 1;
    error = clEnqueueNDRangeKernel(cmdQ, preparePhaseI, 1, nullptr, &count, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &complete);

    E_CL(error, "Q(setup)");
    lastEvent = complete;

    if (verbose) {
        cout << "CL-Solver:: prepared phase I tableau\n";
    }
    if (verbose > 1) {
        showBuffer(tableauBuf0, tableSize[0], tableSize[1]);
    }

    return true;
}

bool CombinedSimplexCLSolver::readObjectiveSolutionValue(double& value, cl_mem readBuf) {
    cl_int error = CL_SUCCESS;

    ++readOps;
    bytesRead += sizeof(double);
    error = clEnqueueReadBuffer(cmdQ, readBuf, CL_TRUE, 0, sizeof(double), &value, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, nullptr);
    E_CL(error, "R(pivot)");
    lastEvent = nullptr;

    return true;
}

bool CombinedSimplexCLSolver::buildSecondPhaseTableau(Matrix& problem, cl_mem readBuf, cl_mem writeBuf) {
    cl_int error = CL_SUCCESS;

    tableSize[0] = problem.rows();
    tableSize[1] = problem.columns() + slacks_p2;
    slacks = slacks_p2;

    partition[0] = std::max(1, (int32_t)std::min((int32_t)CLSys::getInstance().deviceWorkGroupSize(), (int32_t)ceil(sqrt(tableSize[1]) - 1)));
    partition[1] = std::max(1, (int32_t)std::min((int32_t)CLSys::getInstance().deviceWorkGroupSize(), (int32_t)ceil(sqrt(tableSize[0] - 1))));
    scannersNo = std::max(partition[0], partition[1]);
 
    const uint32_t variables = problem.columns() - 1;
    const size_t setupWorkSize[] = { (size_t)tableSize[0], (size_t)tableSize[1] };
    const int32_t objective_sign = minimize ? 1 : -1;
    clSetKernelArg(setupTableauPhaseII, 0, sizeof(cl_mem), &problemBuf);
    clSetKernelArg(setupTableauPhaseII, 1, sizeof(cl_mem), &readBuf);
    clSetKernelArg(setupTableauPhaseII, 2, sizeof(cl_mem), &writeBuf);
    clSetKernelArg(setupTableauPhaseII, 3, sizeof(uint32_t), &variables);
    clSetKernelArg(setupTableauPhaseII, 4, sizeof(uint32_t), &slacks_p1);
    clSetKernelArg(setupTableauPhaseII, 5, sizeof(uint32_t), &objective_sign);

    cl_event complete;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, setupTableauPhaseII, 2, nullptr, setupWorkSize, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &complete);
    E_CL(error, "Q(setup2)");
    lastEvent = complete;

    if (verbose) {
        cout << "CL-Solver:: Phase II: tableau built\n";
    }
    if (verbose > 1) {
        showBuffer(writeBuf, tableSize[0], tableSize[1]);
    }

    return true;
}

bool CombinedSimplexCLSolver::zeroOutBasicSolutionObjective(Matrix& problem, cl_mem readBuf, cl_mem writeBuf) {
    cl_int error = CL_SUCCESS;

    clSetKernelArg(detectBasic, 0, sizeof(cl_mem), &writeBuf);
    clSetKernelArg(detectBasic, 1, sizeof(cl_mem), &indexBuf);
    clSetKernelArg(detectBasic, 2, sizeof(uint32_t) * 2, tableSize);

    size_t count = problem.columns() - 1;
    cl_event detect;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, detectBasic, 1, nullptr, &count, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &detect);
    E_CL(error, "Q(detect)");

    int32_t rows = count;
    clSetKernelArg(zeroOutBasic, 0, sizeof(cl_mem), &readBuf);
    clSetKernelArg(zeroOutBasic, 1, sizeof(cl_mem), &writeBuf);
    clSetKernelArg(zeroOutBasic, 2, sizeof(cl_mem), &indexBuf);
    clSetKernelArg(zeroOutBasic, 3, sizeof(uint32_t) * 2, tableSize);
    clSetKernelArg(zeroOutBasic, 4, sizeof(int32_t), &rows);

    count = tableSize[1];
    cl_event complete;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, zeroOutBasic, 1, nullptr, &count, nullptr, 1, &detect, &complete);
    E_CL(error, "Q(zero-out)");
    lastEvent = complete;

    if (verbose) {
        cout << "CL-Solver:: Phase II: zeroed out basic solution\n";
    }
    if (verbose > 1) {
        showBuffer(writeBuf, tableSize[0], tableSize[1]);
    }

    return true;
}

bool CombinedSimplexCLSolver::runSelectPivot(const Matrix& problem, const cl_mem& tableauBuf) {
    cl_int error = CL_SUCCESS;

    clSetKernelArg(selectPivot, 0, sizeof(cl_mem), &tableauBuf);
    clSetKernelArg(selectPivot, 1, sizeof(cl_mem), &indexBuf);
    clSetKernelArg(selectPivot, 2, sizeof(int32_t) * 2, &tableSize);
    clSetKernelArg(selectPivot, 3, sizeof(int32_t) * 2, &partition);

    cl_event complete;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, selectPivot, 1, nullptr, &scannersNo, &scannersNo, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &complete);
    E_CL(error, "Q(selectPivot)");
    lastEvent = complete;

    ++readOps;
    bytesRead += sizeof(int32_t) * 2;
    int32_t pivot[2];
    error = clEnqueueReadBuffer(cmdQ, indexBuf, CL_TRUE, 0, 2 * sizeof(int32_t), pivot, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, nullptr);
    E_CL(error, "R(pivot)");

    pivotColumn = pivot[0];
    pivotRow = pivot[1];

    if (verbose) {
        cout << "CL-Solver:: pivot column: " << (int)pivotColumn << endl;
        cout << "CL-Solver:: pivot row : " << (int)pivotRow << endl;
    }

    lastEvent = 0;
    return true;
}

bool CombinedSimplexCLSolver::runClearWithPivot(const Matrix& problem, const cl_mem& read, const cl_mem& write) {
    cl_int error = CL_SUCCESS;

    const uint32_t pivot[] = {pivotColumn, pivotRow};
    clSetKernelArg(clearWithPivot, 0, sizeof(cl_mem), &read);
    clSetKernelArg(clearWithPivot, 1, sizeof(cl_mem), &write);
    clSetKernelArg(clearWithPivot, 2, sizeof(uint32_t) * 2, pivot);

    const size_t work_size[] = { (size_t)tableSize[1], (size_t)tableSize[0] };
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, clearWithPivot, 2, nullptr, work_size, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &lastEvent);
    E_CL(error, "Q(clear)");

    return true;
}

bool CombinedSimplexCLSolver::runExtractSolution(const Matrix& problem, const cl_mem& tableau, const cl_mem& solutionBuf) {
#define DIV2_NEXT(x)    \
    (((x) > 1 && (x) % 2) ? 1 + (x) / 2 : (x) / 2)

    cl_int error = CL_SUCCESS;

    clSetKernelArg(buildSolutionTable, 0, sizeof(cl_mem), &tableau);
    clSetKernelArg(buildSolutionTable, 1, sizeof(cl_mem), &solutionBuf);

    size_t work_size[] = { (size_t)tableSize[0], (size_t)tableSize[1] };
    cl_event setup;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, buildSolutionTable, 2, nullptr, work_size, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &setup);
    E_CL(error, "Q(build-sol)");
    lastEvent = 0;

    if (verbose > 2) {
        showBuffer(solutionBuf, tableSize[0], tableSize[1]);
    }

    const uint32_t stride = tableSize[1];
    const uint32_t elementsNo = tableSize[0] * tableSize[1];
    clSetKernelArg(extractSolution, 0, sizeof(cl_mem), &solutionBuf);
    clSetKernelArg(extractSolution, 1, sizeof(uint32_t), &stride);
    clSetKernelArg(extractSolution, 2, sizeof(uint32_t), &elementsNo);
    cl_event previous = setup;
    for (size_t n = DIV2_NEXT(problem.rows()); n > 0; n = DIV2_NEXT(n)) {
        work_size[0] = n;
        cl_event next;

        ++programsRun;
        error = clEnqueueNDRangeKernel(cmdQ, extractSolution, 2, nullptr, work_size, nullptr, 1, &previous, &next);
        E_CL(error, "Q(extract-sol)");
        previous = next;

        if (verbose > 2) {
            cout << "n = " << n << endl;
            showBuffer(solutionBuf, tableSize[0], tableSize[1]);
        }
    }

    clSetKernelArg(fixInfinities, 0, sizeof(cl_mem), &solutionBuf);
    cl_event fixed;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, fixInfinities, 1, nullptr, work_size + 1, nullptr, 1, &previous, &fixed);
    E_CL(error, "Q(fix-inf)");

    ++readOps;
    bytesRead += sizeof(double) * (problem.columns() - 1);
    error = clEnqueueReadBuffer(cmdQ, solutionBuf, CL_TRUE, 0, sizeof(double) * (problem.columns() - 1), lastSolution->rawptr(), 1, &fixed, nullptr);
    E_CL(error, "R(solution-1)");

    double last_value = 0;
    ++readOps;
    bytesRead += sizeof(double);
    // NOTE: read from the initial buffer, not the processed one!
    error = clEnqueueReadBuffer(cmdQ, tableau, CL_TRUE, (elementsNo - 1) * sizeof(double), sizeof(double), &last_value, 1, &fixed, nullptr);
    E_CL(error, "R(solution-2)");

    lastSolution->at(0, lastSolution->columns() - 1) = last_value;
    return true;
}

void CombinedSimplexCLSolver::saveBuffer(const cl_mem& buffer, int rows, int columns) {
    if (!prefix) {
        return;
    }

    char buf[512];
    sprintf(buf, "%s.%u.csv", prefix, steps);

    Matrix tmp(rows, columns);
    ++readOps;
    bytesRead += tmp.size() * sizeof(double);
    cl_int error = clEnqueueReadBuffer(cmdQ, buffer, CL_TRUE, 0, tmp.size() * sizeof(double), tmp.rawptr(), lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, nullptr);
    if (error != CL_SUCCESS) {
        cerr << "CL-Solver[R(buffer)] ERROR: " << clErrorString(error) << endl;
        return;
    }

    tmp.toCSV(buf);
    lastEvent = 0;
}

void CombinedSimplexCLSolver::showBuffer(const cl_mem& buffer, int rows, int columns) {
    Matrix tmp(rows, columns);
    ++readOps;
    bytesRead += tmp.size() * sizeof(double);
    cl_int error = clEnqueueReadBuffer(cmdQ, buffer, CL_TRUE, 0, tmp.size() * sizeof(double), tmp.rawptr(), lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, nullptr);
    if (error != CL_SUCCESS) {
        cerr << "CL-Solver[R(buffer)] ERROR: " << clErrorString(error) << endl;
        return;
    }

    tmp.print();
    lastEvent = 0;
}

void CombinedSimplexCLSolver::showIntBuffer(const cl_mem& buffer, int rows, int columns) {
    int32_t* table = new int32_t[rows * columns];
    ++readOps;
    bytesRead += rows * columns * sizeof(int32_t);
    cl_int error = clEnqueueReadBuffer(cmdQ, buffer, CL_TRUE, 0, rows * columns * sizeof(int32_t), table, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, nullptr);
    if (error != CL_SUCCESS) {
        delete [] table;
        cerr << "CL-Solver[R(buffer)] ERROR: " << clErrorString(error) << endl;
        return;
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) {
            cout << table[r * columns + c] << " ";
        }
        cout << "\n";
    }

    lastEvent = 0;
    delete [] table;
}

