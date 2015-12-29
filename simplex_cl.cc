#include <iostream>
#include <cmath>
#include <sstream>
#include <chrono>
#include <ctime>

#include "clutil.h"
#include "simplex_cl.h"

using namespace std;


#define E_CL(e, where) \
    if ((e) != CL_SUCCESS) { \
        cerr << "CL-Solver[" where "] ERROR(" << (-(e)) << "): " << clErrorString(e) << endl; \
        return false; \
    }

#define DIV2_NEXT(x)    \
    (((x) > 1 && (x) % 2) ? 1 + (x) / 2 : (x) / 2)

SimplexCLSolver::SimplexCLSolver() {
    problemBuf = 0;
    tableauBuf0 = 0;
    tableauBuf1 = 0;
    indexBuf = 0;
    useBinaryDividers = false;

    CLSys& cl = CLSys::getInstance();

    cmdQ = cl.createQueue();
    program = cl.buildProgram(loadTextFile("cl/simplex.cl"));
    setupTableau = cl.createKernel(program, "setupTableau");
    findMin = cl.createKernel(program, "findMin");
    selectNegatives = cl.createKernel(program, "selectNegatives");
    selectMinNegative = cl.createKernel(program, "selectMinNegative");
    selectMinRatio = cl.createKernel(program, "selectMinRatio");
    copyPivotColumnRatios = cl.createKernel(program, "copyPivotColumnRatios");
    findMinRatio = cl.createKernel(program, "findMinRatio");
    clearWithPivot = cl.createKernel(program, "clearWithPivot");
    buildSolutionTable = cl.createKernel(program, "buildSolutionTable");
    extractSolution = cl.createKernel(program, "extractSolution");
    fixInfinities = cl.createKernel(program, "fixInfinities");
}

SimplexCLSolver::~SimplexCLSolver() {
    if (fixInfinities) clReleaseKernel(fixInfinities);
    if (extractSolution) clReleaseKernel(extractSolution);
    if (buildSolutionTable) clReleaseKernel(buildSolutionTable);
    if (clearWithPivot) clReleaseKernel(clearWithPivot);
    if (findMin) clReleaseKernel(findMin);
    if (selectNegatives) clReleaseKernel(selectNegatives);
    if (selectMinNegative) clReleaseKernel(selectMinNegative);
    if (selectMinRatio) clReleaseKernel(selectMinRatio);
    if (findMinRatio) clReleaseKernel(findMinRatio);
    if (copyPivotColumnRatios) clReleaseKernel(copyPivotColumnRatios);
    if (setupTableau) clReleaseKernel(setupTableau);
    if (program) clReleaseProgram(program);
    if (problemBuf) clReleaseMemObject(problemBuf);
    if (tableauBuf0) clReleaseMemObject(tableauBuf0);
    if (tableauBuf1) clReleaseMemObject(tableauBuf1);
    if (indexBuf) clReleaseMemObject(indexBuf);
    if (cmdQ) clReleaseCommandQueue(cmdQ);
}

bool SimplexCLSolver::initOK() const {
    return cmdQ && program && setupTableau && findMin &&
           selectNegatives && copyPivotColumnRatios &&
           findMinRatio && clearWithPivot && buildSolutionTable &&
           selectMinNegative && selectMinRatio &&
           extractSolution && fixInfinities;
}

void SimplexCLSolver::printStats() const {
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

string SimplexCLSolver::report() const {
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

bool SimplexCLSolver::init(Matrix& problem) {
    steps = programsRun = writeOps = readOps = 0;
    bytesWriten = bytesRead = bytesAlloc = 0;

    if (verbose) {
        cout << "CL-Solver:: init\n";
    }

    if (!buildBuffers(problem)) return false;
    if (!runSetup(problem)) return false;
    return true;
}

void SimplexCLSolver::solve(Matrix& problem) {
    if (verbose) {
        cout << "CL-Solver:: begins\n";
    }

    cl_mem doubleBuf[] = { tableauBuf0, tableauBuf1 };
    aborted = true;

    while(steps < maxSteps) {
        cl_mem read_buf = doubleBuf[steps%2];
        cl_mem write_buf = doubleBuf[(steps + 1)%2];
        ++steps;

        if (verbose) {
            cout << "CL-Solver:: entering step: " << steps << endl;
        }

        if (!runSelectPivotColumn(problem, read_buf)) return;
        if (pivotColumn == (uint32_t)-1) {
            saveBuffer(write_buf, problem.rows(), problem.rows() + problem.columns());

            --steps;
            aborted = false;
            break; // no pivot? we found a solution :)
        }

        if (!runSelectPivotRow(problem, read_buf)) return;
        if (pivotRow == (uint32_t)-1) {
            saveBuffer(write_buf, problem.rows(), problem.rows() + problem.columns());

            --steps;
            noSolution = true;
            aborted = false;
            break; // no pivot? we found a solution :)
        }

        if (!runClearWithPivot(problem, read_buf, write_buf)) return;
        
        saveBuffer(write_buf, problem.rows(), problem.rows() + problem.columns());

        if (verbose > 1) {
            showBuffer(write_buf, problem.rows(), problem.rows() + problem.columns());
        }
    }

    if (verbose) {
        cout << "CL-Solver:: compiling solution\n";
    }

    prepareNewSolution(problem.columns());
    if (!aborted && !noSolution) {
        if (!runExtractSolution(problem, doubleBuf[steps%2], doubleBuf[(steps + 1)%2])) return;
    }

    if (verbose) {
        cout << "CL-Solver:: completed\n";
    }
}

bool SimplexCLSolver::buildBuffers(Matrix& problem) {
    CLSys& cl = CLSys::getInstance();

    cl_int error = CL_SUCCESS;
    bytesAlloc += problem.size() * sizeof(double);
    writeOps++;
    bytesWriten += problem.size() * sizeof(double);
    problemBuf = clCreateBuffer(cl.ctx(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, problem.size() * sizeof(double), problem.rawptr(), &error);
    E_CL(error, "C(problem)");

    uint32_t tableauBufSize = problem.rows() * (problem.rows() + problem.columns()) * sizeof(double);
    bytesAlloc += tableauBufSize;
    tableauBuf0 = clCreateBuffer(cl.ctx(), CL_MEM_READ_WRITE, tableauBufSize, 0, &error);
    E_CL(error, "C(tableau-0)");

    bytesAlloc += tableauBufSize;
    tableauBuf1 = clCreateBuffer(cl.ctx(), CL_MEM_READ_WRITE, tableauBufSize, 0, &error);
    E_CL(error, "C(tableau-1)");

    bytesAlloc += sizeof(uint32_t) * (problem.rows() + problem.columns());
    indexBuf = clCreateBuffer(cl.ctx(), CL_MEM_READ_WRITE, sizeof(uint32_t) * (problem.rows() + problem.columns()), 0, &error);
    E_CL(error, "C(index)");

    return true;
}

bool SimplexCLSolver::runSetup(const Matrix& problem) {
    cl_int error = CL_SUCCESS;

    const uint32_t variables = problem.columns();
    clSetKernelArg(setupTableau, 0, sizeof(cl_mem), &problemBuf);
    clSetKernelArg(setupTableau, 1, sizeof(cl_mem), &tableauBuf0);
    clSetKernelArg(setupTableau, 2, sizeof(uint32_t), &variables);

    const size_t setupWorkSize[] = { problem.rows(), problem.rows() + problem.columns() };
    cl_event setupComplete;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, setupTableau, 2, nullptr, setupWorkSize, nullptr, 0, nullptr, &setupComplete);
    E_CL(error, "Q(setup)");
    lastEvent = setupComplete;

    if (verbose > 1) {
        showBuffer(tableauBuf0, problem.rows(), problem.rows() + problem.columns());
    }

    return true;
}

bool SimplexCLSolver::runSelectPivotColumn(const Matrix& problem, const cl_mem& tableauBuf) {
    if (useBinaryDividers) {
        if (!runSelectPivotColumnBinaryDiv(problem, tableauBuf)) {
            return false;
        }
    } else {
        if (!runSelectPivotColumnItr(problem, tableauBuf)) {
            return false;
        }
    }

    ++readOps;
    bytesRead += sizeof(uint32_t);
    cl_int error = clEnqueueReadBuffer(cmdQ, indexBuf, CL_TRUE, 0, sizeof(uint32_t), &pivotColumn, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, nullptr);
    E_CL(error, "R(pivot-col)");

    if (verbose) {
        cout << "CL-Solver:: pivot column: " << (int)pivotColumn << endl;
    }

    lastEvent = 0;
    return true;
}

bool SimplexCLSolver::runSelectPivotColumnBinaryDiv(const Matrix& problem, const cl_mem& tableauBuf) {
    cl_int error = CL_SUCCESS;

    const uint32_t offset = (problem.rows() - 1) * (problem.columns() + problem.rows());
    clSetKernelArg(selectNegatives, 0, sizeof(cl_mem), &tableauBuf);
    clSetKernelArg(selectNegatives, 1, sizeof(cl_mem), &indexBuf);
    clSetKernelArg(selectNegatives, 2, sizeof(uint32_t), &offset);

    const size_t maxIndex = problem.rows() + problem.columns() - 1;

    cl_event init_step;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, selectNegatives, 1, nullptr, &maxIndex, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &init_step);
    E_CL(error, "Q(select<0)");

    if (verbose > 2) {
        lastEvent = init_step;
        showIntBuffer(indexBuf, 1, maxIndex);
    }

    clSetKernelArg(findMin, 0, sizeof(cl_mem), &tableauBuf);
    clSetKernelArg(findMin, 1, sizeof(cl_mem), &indexBuf);
    clSetKernelArg(findMin, 3, sizeof(uint32_t), &maxIndex);
    clSetKernelArg(findMin, 4, sizeof(uint32_t), &offset);
    cl_event previous = init_step;
    for (size_t n = DIV2_NEXT(maxIndex); n > 0; n = DIV2_NEXT(n)) {
        clSetKernelArg(findMin, 2, sizeof(uint32_t), &n);
        cl_event next;

        ++programsRun;
        error = clEnqueueNDRangeKernel(cmdQ, findMin, 1, nullptr, &n, nullptr, 1, &previous, &next);
        E_CL(error, "Q(findMin)");
        previous = next;

        if (verbose > 2) {
            cout << "n = " << n << endl;
            showIntBuffer(indexBuf, 1, maxIndex);
        }
    }

    lastEvent = init_step;
    return true;
}

bool SimplexCLSolver::runSelectPivotColumnItr(const Matrix& problem, const cl_mem& tableauBuf) {
    cl_int error = CL_SUCCESS;

    const size_t work_size = (size_t)ceil(1 + sqrt(problem.columns() + problem.rows()));
    clSetKernelArg(selectMinNegative, 0, sizeof(cl_mem), &tableauBuf);
    clSetKernelArg(selectMinNegative, 1, sizeof(cl_mem), &indexBuf);
    const uint32_t offset = (problem.rows() - 1) * (problem.columns() + problem.rows());
    const uint32_t count = problem.columns() + problem.rows() - 1;
    clSetKernelArg(selectMinNegative, 2, sizeof(uint32_t), &offset);
    clSetKernelArg(selectMinNegative, 3, sizeof(uint32_t), &count);

    cl_event init_step;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, selectMinNegative, 1, nullptr, &work_size, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &init_step);
    E_CL(error, "Q(selectMin<0)");

    lastEvent = init_step;
    if (verbose > 2) {
        showIntBuffer(indexBuf, 1, 1);
    }

    return true;
}

bool SimplexCLSolver::runSelectPivotRow(const Matrix& problem, const cl_mem& tableauBuf) {
    if (useBinaryDividers) {
        if (!runSelectPivotRowBinaryDiv(problem, tableauBuf)) {
            return false;
        }
    } else {
        if (!runSelectPivotRowItr(problem, tableauBuf)) {
            return false;
        }
    }

    ++readOps;
    bytesRead += sizeof(uint32_t);
    cl_int error = clEnqueueReadBuffer(cmdQ, indexBuf, CL_TRUE, 0, sizeof(uint32_t), &pivotRow, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, nullptr);
    E_CL(error, "R(pivot-row)");

    if (verbose) {
        cout << "CL-Solver:: pivot row: " << pivotRow << endl;
    }

    lastEvent = 0;
    return true;
}

bool SimplexCLSolver::runSelectPivotRowBinaryDiv(const Matrix& problem, const cl_mem& tableauBuf) {
    cl_int error = CL_SUCCESS;

    const uint32_t offset[] = { pivotColumn, problem.rows() + problem.columns() - 1 };
    const uint32_t stride = offset[1] + 1;
    const uint32_t otherOffset = DIV2_NEXT(problem.rows());
    const uint32_t maxIndex = problem.rows() - 1;
    clSetKernelArg(copyPivotColumnRatios, 0, sizeof(cl_mem), &tableauBuf);
    clSetKernelArg(copyPivotColumnRatios, 1, sizeof(cl_mem), &indexBuf);
    clSetKernelArg(copyPivotColumnRatios, 2, sizeof(uint32_t) * 2, offset);
    clSetKernelArg(copyPivotColumnRatios, 3, sizeof(uint32_t), &stride);
    clSetKernelArg(copyPivotColumnRatios, 4, sizeof(uint32_t), &otherOffset);
    clSetKernelArg(copyPivotColumnRatios, 5, sizeof(uint32_t), &maxIndex);

    const size_t count = otherOffset;
    cl_event init_step;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, copyPivotColumnRatios, 1, nullptr, &count, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &init_step);
    E_CL(error, "R(copyRatio)");

    if (verbose > 2) {
        lastEvent = init_step;
        showIntBuffer(indexBuf, 1, maxIndex);
    }

    clSetKernelArg(findMinRatio, 0, sizeof(cl_mem), &tableauBuf);
    clSetKernelArg(findMinRatio, 1, sizeof(cl_mem), &indexBuf);
    clSetKernelArg(findMinRatio, 2, sizeof(uint32_t) * 2, offset);
    clSetKernelArg(findMinRatio, 3, sizeof(uint32_t), &stride);
    clSetKernelArg(findMinRatio, 5, sizeof(uint32_t), &count);
    cl_event previous = init_step;
    for (size_t n = DIV2_NEXT(count); n > 0; n = DIV2_NEXT(n)) {
        clSetKernelArg(findMinRatio, 4, sizeof(uint32_t), &n);
        cl_event next;

        ++programsRun;
        error = clEnqueueNDRangeKernel(cmdQ, findMinRatio, 1, nullptr, &n, nullptr, 1, &previous, &next);
        E_CL(error, "Q(findMinRatio)");
        previous = next;

        if (verbose > 2) {
            cout << "n = " << n << endl;
            showIntBuffer(indexBuf, 1, maxIndex);
        }
    }

    lastEvent = previous;
    return true;
}

bool SimplexCLSolver::runSelectPivotRowItr(const Matrix& problem, const cl_mem& tableauBuf) {
    cl_int error = CL_SUCCESS;

    clSetKernelArg(selectMinRatio, 0, sizeof(cl_mem), &tableauBuf);
    const uint32_t offset = pivotColumn;
    const uint32_t count = problem.rows() - 1;
    const uint32_t stride = problem.rows() + problem.columns();
    const uint32_t otherOffset = stride - 1;
    clSetKernelArg(selectMinRatio, 1, sizeof(cl_mem), &indexBuf);
    clSetKernelArg(selectMinRatio, 2, sizeof(uint32_t), &offset);
    clSetKernelArg(selectMinRatio, 3, sizeof(uint32_t), &count);
    clSetKernelArg(selectMinRatio, 4, sizeof(uint32_t), &stride);
    clSetKernelArg(selectMinRatio, 5, sizeof(uint32_t), &otherOffset);

    const size_t work_size = (size_t)ceil(sqrt(problem.rows()) + 1);
    cl_event init_step;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, selectMinRatio, 1, nullptr, &work_size, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &init_step);
    E_CL(error, "Q(selectMinRatio)");

    lastEvent = init_step;
    return true;
}

bool SimplexCLSolver::runClearWithPivot(const Matrix& problem, const cl_mem& read, const cl_mem& write) {
    cl_int error = CL_SUCCESS;

    const uint32_t pivot[] = {pivotColumn, pivotRow};
    const uint32_t stride = problem.rows() + problem.columns();
    clSetKernelArg(clearWithPivot, 0, sizeof(cl_mem), &read);
    clSetKernelArg(clearWithPivot, 1, sizeof(cl_mem), &write);
    clSetKernelArg(clearWithPivot, 2, sizeof(uint32_t) * 2, pivot);
    clSetKernelArg(clearWithPivot, 3, sizeof(uint32_t), &stride);

    const size_t work_size[] = {problem.columns() + problem.rows(), problem.rows()};
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, clearWithPivot, 2, nullptr, work_size, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &lastEvent);
    E_CL(error, "Q(clear)");

    return true;
}

bool SimplexCLSolver::runExtractSolution(const Matrix& problem, const cl_mem& tableau, const cl_mem& solutionBuf) {
    cl_int error = CL_SUCCESS;

    const uint32_t stride = problem.rows() + problem.columns();
    clSetKernelArg(buildSolutionTable, 0, sizeof(cl_mem), &tableau);
    clSetKernelArg(buildSolutionTable, 1, sizeof(cl_mem), &solutionBuf);
    clSetKernelArg(buildSolutionTable, 2, sizeof(uint32_t), &stride);

    size_t work_size[] = {problem.rows(), problem.columns() - 1};
    cl_event setup;
    ++programsRun;
    error = clEnqueueNDRangeKernel(cmdQ, buildSolutionTable, 2, nullptr, work_size, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &setup);
    E_CL(error, "Q(build-sol)");
    lastEvent = 0;

    if (verbose > 2) {
        showBuffer(solutionBuf, problem.rows(), problem.rows() + problem.columns());
    }

    const uint32_t maxIndex = problem.rows() * (problem.rows() + problem.columns());
    clSetKernelArg(extractSolution, 0, sizeof(cl_mem), &solutionBuf);
    clSetKernelArg(extractSolution, 1, sizeof(uint32_t), &stride);
    clSetKernelArg(extractSolution, 2, sizeof(uint32_t), &maxIndex);
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
            showBuffer(solutionBuf, problem.rows(), problem.rows() + problem.columns());
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

    double last2values[] = {0,0};
    // NOTE: read from the initial buffer, not the processed one!
    ++readOps;
    bytesRead += sizeof(double) * 2;
    error = clEnqueueReadBuffer(cmdQ, tableau, CL_TRUE, (problem.rows() * (problem.rows() + problem.columns()) - 2) * sizeof(double), sizeof(double) * 2, last2values, 1, &fixed, nullptr);
    E_CL(error, "R(solution-2)");

    lastSolution->at(0, lastSolution->columns() - 1) = fabs(last2values[0]) > 1e-20 ? last2values[1] / last2values[0] : 0;
    return true;
}

void SimplexCLSolver::saveBuffer(const cl_mem& buffer, int rows, int columns) {
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

void SimplexCLSolver::showBuffer(const cl_mem& buffer, int rows, int columns) {
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

void SimplexCLSolver::showIntBuffer(const cl_mem& buffer, int rows, int columns) {
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

