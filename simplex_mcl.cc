#include <iostream>
#include <cmath>
#include <sstream>
#include <chrono>
#include <ctime>

#include "clutil.h"
#include "simplex_mcl.h"

using namespace std;


#define E_CL(e, where) \
    if ((e) != CL_SUCCESS) { \
        cerr << "MCL-Solver[" where "] ERROR(" << (-(e)) << "): " << clErrorString(e) << endl; \
        return false; \
    }

MonolithicSimplexCLSolver::MonolithicSimplexCLSolver() {
    problemBuf = 0;
    tableauBuf0 = 0;
    tableauBuf1 = 0;
    indexBuf = 0;

    CLSys& cl = CLSys::getInstance();

    cmdQ = cl.createQueue();
    program = cl.buildProgram(loadTextFile("cl/monolithic.cl"));
    setupTableau = cl.createKernel(program, "setupTableau");
    simplex = cl.createKernel(program, "monolithicSimplex");
}

MonolithicSimplexCLSolver::~MonolithicSimplexCLSolver() {
    if (simplex) clReleaseKernel(simplex);
    if (setupTableau) clReleaseKernel(setupTableau);
    if (program) clReleaseProgram(program);
    if (problemBuf) clReleaseMemObject(problemBuf);
    if (tableauBuf0) clReleaseMemObject(tableauBuf0);
    if (tableauBuf1) clReleaseMemObject(tableauBuf1);
    if (indexBuf) clReleaseMemObject(indexBuf);
    if (cmdQ) clReleaseCommandQueue(cmdQ);
}

bool MonolithicSimplexCLSolver::initOK() const {
    return cmdQ && program && setupTableau && simplex;
}

void MonolithicSimplexCLSolver::printStats() const {
    cout << "Execution Statistics:\n  Iterations: " << steps <<
        "\n  Aborted: " << (aborted?"true":"false") <<
        "\n  CL Programs invocations: 2" << 
        "\n  CL Write ops: 1" <<
        "\n  CL Read ops: 2" <<
        "\n  CL Allocation (bytes): " << bytesAlloc <<
        "\n  CL Writen (bytes): " << bytesWriten <<
        "\n  CL Read (bytes): " << bytesRead <<
        endl;
}

string MonolithicSimplexCLSolver::report() const {
    stringstream ss;
    ss << steps << "," << (aborted?1:0);
    ss << ":";
    ss << bytesAlloc << ",";
    ss << bytesWriten << ",";
    ss << bytesRead;
    return ss.str();
}

bool MonolithicSimplexCLSolver::init(Matrix& problem) {
    steps = 0;
    bytesWriten = bytesRead = bytesAlloc = 0;

    if (verbose) {
        cout << "MCL-Solver:: init\n";
    }

    if (!buildBuffers(problem)) return false;
    if (!runSetup(problem)) return false;

    return true;
}

void MonolithicSimplexCLSolver::solve(Matrix& problem) {
    if (verbose) {
        cout << "MCL-Solver:: begins (itr max: " << maxSteps << ")\n";
    }

    if (verbose) {
        cout << "MCL-Solver:: entering monolithic GPU solver\n";
    }

    if (!runSimplex(problem)) return;

    if (verbose) {
        cout << "MCL-Solver:: fetching solution\n";
    }

    prepareNewSolution(problem.columns());
    if (!extractSolution(problem)) return;

    if (verbose > 1) {
        cout << "buf" << (steps%2) << "\n";
        showBuffer(steps%2 == 0 ? tableauBuf0 : tableauBuf1, problem.rows(), problem.rows() + problem.columns());
        if (verbose > 2) {
            cout << "buf" << ((steps+1)%2) << "\n";
            showBuffer(steps%2 == 1 ? tableauBuf0 : tableauBuf1, problem.rows(), problem.rows() + problem.columns());
        }
    }

    if (verbose) {
        cout << "MCL-Solver:: completed\n";
    }
}

bool MonolithicSimplexCLSolver::buildBuffers(Matrix& problem) {
    CLSys& cl = CLSys::getInstance();

    cl_int error = CL_SUCCESS;
    bytesAlloc += problem.size() * sizeof(double);
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

    rowPartition = std::max(1u, (uint32_t)ceil(sqrt(problem.columns() + problem.rows()) - 2));
    colPartition = std::max(1u, (uint32_t)ceil(sqrt(problem.rows() - 1)));
    uint32_t indexBufSz = std::max(6u, std::max(rowPartition, colPartition) + 1);

    if (verbose) {
        cout << "MCL-Solver:: partitions: R=" << rowPartition << " / C=" << colPartition << endl;
    }

    bytesAlloc += sizeof(int32_t) * indexBufSz;
    indexBuf = clCreateBuffer(cl.ctx(), CL_MEM_READ_WRITE, sizeof(int32_t) * indexBufSz, 0, &error);
    E_CL(error, "C(index)");

    return true;
}

bool MonolithicSimplexCLSolver::runSetup(const Matrix& problem) {
    cl_int error = CL_SUCCESS;

    const uint32_t variables = problem.columns();
    clSetKernelArg(setupTableau, 0, sizeof(cl_mem), &problemBuf);
    clSetKernelArg(setupTableau, 1, sizeof(cl_mem), &tableauBuf0);
    clSetKernelArg(setupTableau, 2, sizeof(uint32_t), &variables);

    const size_t setupWorkSize[] = { problem.rows(), problem.rows() + problem.columns() };
    cl_event setupComplete;
    error = clEnqueueNDRangeKernel(cmdQ, setupTableau, 2, nullptr, setupWorkSize, nullptr, 0, nullptr, &setupComplete);
    E_CL(error, "Q(setup)");
    lastEvent = setupComplete;

    if (verbose > 1) {
        showBuffer(tableauBuf0, problem.rows(), problem.rows() + problem.columns());
    }

    return true;
}

bool MonolithicSimplexCLSolver::runSimplex(const Matrix& problem) {
    cl_int error = CL_SUCCESS;

    const uint32_t variables = problem.columns() - 1;
    clSetKernelArg(simplex, 0, sizeof(cl_mem), &tableauBuf0);
    clSetKernelArg(simplex, 1, sizeof(cl_mem), &tableauBuf1);
    clSetKernelArg(simplex, 2, sizeof(cl_mem), &indexBuf);
    clSetKernelArg(simplex, 3, sizeof(uint32_t), &maxSteps);
    clSetKernelArg(simplex, 4, sizeof(uint32_t), &rowPartition);
    clSetKernelArg(simplex, 5, sizeof(uint32_t), &colPartition);
    clSetKernelArg(simplex, 6, sizeof(uint32_t), &variables);

    const size_t workSize[] = { problem.rows(), problem.rows() + problem.columns() };
    cl_event simplex_complete;
    error = clEnqueueNDRangeKernel(cmdQ, simplex, 2, nullptr, workSize, nullptr, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, &simplex_complete);
    E_CL(error, "Q(simplex)");
    lastEvent = simplex_complete;

    return true;
}

bool MonolithicSimplexCLSolver::extractSolution(const Matrix& problem) {
    cl_int error = CL_SUCCESS;
    int32_t steps_abort_pivot[6];
    error = clEnqueueReadBuffer(cmdQ, indexBuf, CL_TRUE, 0, 6 * sizeof(int32_t), steps_abort_pivot, lastEvent ? 1 : 0, lastEvent ? &lastEvent : 0, nullptr);
    E_CL(error, "R(steps/abort)");
    lastEvent = nullptr;
    steps = steps_abort_pivot[0];
    aborted = steps_abort_pivot[1] == 1;

    if (verbose) {
        int32_t* lp = steps_abort_pivot + 2;
        int32_t* p = steps_abort_pivot + 4;
        cout << "MCL-Solver:: last pivot @ [" << lp[0] << ", " << lp[1] << "] / [" << p[0] << ", " << p[1] << "]\n";
    }

    if (aborted) {
        return true;
    }

    cl_mem solutionBuf = steps%2 == 0 ? tableauBuf1 : tableauBuf0;

    bytesRead += sizeof(double) * problem.columns();
    error = clEnqueueReadBuffer(cmdQ, solutionBuf, CL_TRUE, 0, sizeof(double) * problem.columns(), lastSolution->rawptr(), 0, nullptr, nullptr);
    E_CL(error, "R(solution)");

    return true;
}

void MonolithicSimplexCLSolver::showBuffer(const cl_mem& buffer, int rows, int columns) {
    Matrix tmp(rows, columns);
    cl_int error = clEnqueueReadBuffer(cmdQ, buffer, CL_TRUE, 0, tmp.size() * sizeof(double), tmp.rawptr(), 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        cerr << "MCL-Solver[R(buffer)] ERROR: " << clErrorString(error) << endl;
        return;
    }

    tmp.print();
    lastEvent = 0;
}

