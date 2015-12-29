#include <iostream>
#include <fstream>
#include <iterator>
#include <cstring>
#include <unistd.h>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <sstream>

#include "main.h"
#include "matrix.h"
#include "simplex.h"
#include "simplex_cpu.h"
#include "simplex_cl.h"
#include "simplex_mcl.h"
#include "simplex_wg_mcl.h"
#include "simplex_comb_cl.h"
#include "clutil.h"

using namespace std;

int verbose = 0;

bool parse_args(int argc, char** argv);
void help(const char* progname);

static Matrix* problem = nullptr;
static Solver* solver = nullptr;
static char* outputFilename = nullptr;
static bool showExecStats = false;
static bool tabReport = false;
static bool solutionReport = false;

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) {
        help(argv[0]);
        return 1;
    }
    if (!problem) {
        return 0; // help or sth, no need to solve anything
    }

    if (verbose > 1) {
        problem->print();
    }

    auto begin_init = chrono::high_resolution_clock::now();
    if (!solver->init(*problem)) {
        cerr << "solver init failed" << endl;
        return 1;
    }

    auto begin = chrono::high_resolution_clock::now();
    solver->solve(*problem);

    auto end = chrono::high_resolution_clock::now();
    long run_ms = (end - begin).count() / 1000;
    long total_ms = (end - begin_init).count() / 1000;

    if (verbose) {
        cout << "Solution found:\n";
        solver->getLastSolution()->print();
    }

    solver->getLastSolution()->toCSV(outputFilename);

    if (showExecStats || verbose) {
        cout << "Algorithm execution time: " << run_ms << "ms" << endl;
        solver->printStats();
    }

    if (tabReport) {
        cout << run_ms << "," << total_ms << ":" << solver->report();
        if (solutionReport) {
            cout << ":";
            solver->getLastSolution()->print();
        } else {
            cout << endl;
        }
    }

    delete solver;
    delete problem;
    return 0;
}

string loadTextFile(const char* filename) {
    ifstream input(filename);
    string result ((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
    return result;
}


bool loadProblem(const char* problem_f, const char* constr_f, const char* objective_f) {
    if (problem_f) {
        if (constr_f || objective_f) {
            cerr << "both full problem and constraints/objective provided, choose what you want...\n";
            return false;
        }
        if (verbose) {
            cout << "Loading problem from: " << problem_f << endl;
        }
        problem = Matrix::loadCSV(problem_f);
        if (problem->rows() < 3
            || problem->at(problem->rows() - 1, problem->columns() - 1)) {
            cerr << "No objective function in problem definition\n";
            return false;
        }
        return true;
    }

    if (!constr_f|| !objective_f) {
        cerr << "both constraints and objective are need to construct a problem\n";
        return false;
    }

    if (verbose) {
        cout << "Loading constraints from: " << constr_f << endl;
    }
    Matrix* constraintsDef = Matrix::loadCSV(constr_f);

    if (verbose) {
        cout << "Loading objective from: " << objective_f << endl;
    }
    Matrix* objectiveDef = Matrix::loadCSV(objective_f);

    problem = new Matrix(constraintsDef->rows() + 1, constraintsDef->columns());
    problem->copy(*constraintsDef);
    problem->copy(*objectiveDef, constraintsDef->rows());

    delete constraintsDef;
    delete objectiveDef;
    return true;
}


bool parseInt(const std::string& str, long& integer) {
    char* p_end = nullptr;
    integer = strtol(str.c_str(), &p_end, 10);
    return str.length() > 0 && p_end == str.c_str() + str.length();
}

bool parseFloat(const std::string& str, float& number) {
    char* p_end = nullptr;
    number = strtof(str.c_str(), &p_end);
    return str.length() > 0 && p_end == str.c_str() + str.length();
}

bool parseConstraintCounts(const char* counts_spec, int* counts) {
    stringstream spec(counts_spec);
    string opt;
    int n = 0;
    while (getline(spec, opt, ',')) {
        if (n >= 3) {
            cerr << "bad constraint counts spec [count]\n";
            return false;
        }
        long v = 0;
        if (!parseInt(opt, v)) {
            cerr << "bad constraint counts spec [format]\n";
            return false;
        }
        counts[n++] = v;
    }
    if (n != 3) {
        cerr << "bad constraint counts spec [count]\n";
        return false;
    }
    return true;
}

bool generateProblem(const char* options_spec) {
    stringstream spec(options_spec);
    vector<string> opts;
    string opt;
    while (getline(spec, opt, ',')) {
        opts.push_back(opt);
    }

    if (opts.size() < 3 || opts.size() > 6) {
        cerr << "bad spec options count\n";
        return false;
    }

    long vars = 0;
    if (!parseInt(opts[0], vars) || vars < 2) {
        cerr << "bad variables count\n";
        return false;
    }

    long constraints = 0;
    if (!parseInt(opts[1], constraints) || constraints < 2) {
        cerr << "bad constraint count\n";
        return false;
    }

    float maxGenAbsolute = 1000;
    if (!parseFloat(opts[2], maxGenAbsolute) || maxGenAbsolute < 1) {
        cerr << "bad absolute maximum at generation: floating point expected\n";
        return false;
    }

    bool integer_problem = false;
    if (opts.size() > 3) {
        if (opts[3].length() != 1) {
            cerr << "bad numeric contraint spec (not single character)\n";
            return false;
        }
        integer_problem = opts[3][0] == 'I';
    }

    float zero_ratio = 0.1;
    if (opts.size() > 4) {
        if (!parseFloat(opts[4], zero_ratio) || zero_ratio < 0 || zero_ratio >= 1) {
            cerr << "bad zero ratio: [0,1) float expected\n";
            return false;
        }
    }

    long seed = 0;
    if (opts.size() > 5) {
        if (!parseInt(opts[5], seed)) {
            cerr << "bad seed spec\n";
            return false;
        }
    }

    auto begin = chrono::high_resolution_clock::now();
    problem = Matrix::makeRandom(seed, constraints + 1, vars + 1, maxGenAbsolute, integer_problem, zero_ratio, false, false);
    problem->at(constraints, vars) = 0;
    auto end = chrono::high_resolution_clock::now();

    if (showExecStats || verbose) {
        cout << "Problem generation time: " << ((end - begin).count() / 1000) << "ms" << endl;
    }
    
    return true;
}


bool parse_args(int argc, char** argv) {
    CLSys& cl = CLSys::getInstance();
    char* prob_file = nullptr;
    char* constr_file = nullptr;
    char* objctv_file = nullptr;
    char* random_spec = nullptr;
    char* constr_spec = nullptr;
    bool useCL = false;
    bool showCL = false;
    int clPlatform = 0;
    int clDevice = 0;
    int32_t max_itr = -1;
    char* savePrefix = nullptr;
    bool useBinDiv = false;
    bool useMonolithic = false;
    bool useMonolithicWG = false;
    bool useCombined = false;
    bool minimize = false;
    bool force_ph1 = false;
    int opt;

    while ((opt = getopt(argc, argv, "R:P:C:O:o:vd:p:M:gGX:shwWS:bmkcU:q2")) != -1) {
        switch (opt) {
            case '2':
                force_ph1 = true;
                break;
            case 'R':
                random_spec = optarg;
                break;
            case 'P':
                prob_file = strdup(optarg);
                break;
            case 'C':
                constr_file = strdup(optarg);
                break;
            case 'O':        
                objctv_file = strdup(optarg);
                break;
            case 'o':
                outputFilename = strdup(optarg);
                break;
            case 'U':
                constr_spec = strdup(optarg);
                break;
            case 'v':
                ++verbose;
                break;
            case 'd':
                useCL = true;
                clDevice = atoi(optarg);
                break;
            case 'p':
                useCL = true;
                clPlatform = atoi(optarg);
                break;
            case 'M':
                max_itr = atoi(optarg);
                break;
            case 'X':
                useCL = true;
                cl.setDefaultCompilerFlags(optarg);
                break;
            case 'G':
                useCL = true;
                showCL = true;
                break;
            case 'g':
                useCL = true;
                break;
            case 'S':
                savePrefix = strdup(optarg);
                break;
            case 'b':
                useBinDiv = true;
                break;
            case 'm':
                useMonolithic = true;
                break;
            case 'k':
                useMonolithicWG = true;
                break;
            case 'c':
                useCombined = true;
                break;
            case 's':
                showExecStats = true;
                break;
            case 'q':
                minimize = true;
                break;
            case 'W':
                solutionReport = true;
                /* fall-through */
            case 'w':
                tabReport = true;
                break;
            case 'h':
                help(argv[0]);
                return true;
            default:
                cerr << "unknown argument: " << opt << endl;
                return false;
        }
    }

    if (useCL) {
        atexit(CLSys::destroy);
        cl.setup();
        if (showCL) {
            cl.showSystem();
            return true;
        }
    }

    if (!outputFilename) {
        cerr << "output filename required\n";
        return false;
    }

    if (random_spec) {
        if (prob_file || constr_file || objctv_file) {
            cerr << "cannot handle both a concrete problem and a random generation of one!" << endl;
            return false;
        }
        if (!generateProblem(random_spec)) {
            return false;
        }
    } else if (!loadProblem(prob_file, constr_file, objctv_file)) {
        return false;
    }

    if (useCL) {
        if (!cl.initDevice(clPlatform, clDevice)) {
            return false;
        }
        if (useCombined) {
            CombinedSimplexCLSolver* s = new CombinedSimplexCLSolver;
            if (!s->initOK()) {
                return false;
            }
            solver = s;
        } else if (useMonolithic) {
            MonolithicSimplexCLSolver* s = new MonolithicSimplexCLSolver;
            if (!s->initOK()) {
                return false;
            }
            solver = s;
        } else if (useMonolithicWG) {
            MonolithicWorkgroupSimplexCLSolver* s = new MonolithicWorkgroupSimplexCLSolver;
            if (!s->initOK()) {
                return false;
            }
            solver = s;
        } else {
            SimplexCLSolver* s = new SimplexCLSolver;
            s->setUseBinaryDiv(useBinDiv);
            if (!s->initOK()) {
                return false;
            }
            solver = s;
        }
    } else {
        solver = new SimplexCPUSolver;
    }

    if (max_itr >= 0) {
        solver->setMaxIterations(max_itr);
    }

    if (savePrefix && solver) {
        solver->setTableauSavePrefix(savePrefix);
    }

    solver->setMinimize(minimize);
    solver->setForcePhaseI(force_ph1);

    if (constr_spec) {
        int n[3];
        if (!parseConstraintCounts(constr_spec, n)) {
            return false;
        }
        if (n[0] + n[1] + n[2] != (int)problem->rows() - 1) {
            cerr << "Bad sum of constraint type counts, expected: " << (problem->rows() - 1) << "\n";
            return false;
        }
        solver->setConstraintTypeCounts(n[0], n[1], n[2]);
        free(constr_spec);
    } else {
        solver->setConstraintTypeCounts(problem->rows() - 1, 0, 0);
    }

    return true;
}

void help(const char* progname) {
    cout <<
        progname << " usage:\n"
        " LP options:\n"
        "  -R S\tgenerate a random problem. S is comma separated:\n"
        "      \t<var#>,<contraint#>,<abs-max>[,<I|F>[,<zero%>[,<seed>]]]\n"
        "  -P f\tspecify problem CSV file (constraints & objective)\n"
        "  -C f\tspecify constraints CSV file\n"
        "  -O f\tspecify objective CSV file\n"
        "  -w  \twrite stats in tabular form\n"
        "  -W  \twrite the solution and stats in tabular form\n"
        "  -S f\twrite all intermediate steps in CSV files with the given prefix\n"
        "  -U S\tspecifiy constraint types counts (LE,GE,EQ)\n"
        "  -2  \tforce two phase algorithm (even if not necessary)\n"
        "  -q  \tminimize (default is maximize)\n"
        " CL options:\n"
        "  -G  \tshow OpenCL capabilities & exit\n"
        "  -g  \tuse OpenCL solver (default: false)\n"
        "  -p N\tselect the N'th OpenCL platform (base 0)\n"
        "  -d N\tselect the N'th OpenCL device (base 0)\n"
        "  -X s\tpass compiler flags (s string, as is)\n"
        "  -b  \tuse binary search like algorithm in minimums extraction (slower)\n"
        "  -m  \tuse monolithic solver (you don't want this one...)\n"
        "  -k  \tuse workgroup aware monolithic solver\n"
        "  -c  \tuse pivot combined solver (minimizes host intervention)\n"
        " Misc options:\n"
        "  -M N\tconstraint the solver to running no more than N iterations\n"
        "  -o f\tspecify output CSV file\n"
        "  -s  \tshow statistics at end of execution\n"
        "  -v  \tset verbose output (can be used multiple times)\n"
        "  -h  \tshow this and exit\n"
    ;
}

