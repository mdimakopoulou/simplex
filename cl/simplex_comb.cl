
#define EPSILON     1e-20


__kernel void setupTableau(
            global double* problem,
            global double* tableau,
            const uint variables,
            const int objective_sign)
{
    int rows = get_global_size(0);
    int columns = get_global_size(1);

    int row = get_global_id(0);
    int col = get_global_id(1);
    int offt = row * columns + col;

    if (col < variables) {
        if (row < rows - 1) {
            tableau[offt] = problem[row * (variables + 1) + col];
        } else {
            tableau[offt] = objective_sign * problem[row * (variables + 1) + col];
        }
    }
    else if (col < columns - 1) {
        tableau[offt] = row == col - variables ? 1 : 0;
    }
    else {
        tableau[offt] = problem[row * (variables + 1) + variables];
    }
}


__kernel void setupTableauPhaseI(
            global double* problem,
            global double* tableau,
            const uint variables)
{
    int rows = get_global_size(0);
    int columns = get_global_size(1);

    int row = get_global_id(0);
    int col = get_global_id(1);
    int offt = row * columns + col;

    if (col < variables) {
        if (row < rows - 1) {
            tableau[offt] = problem[row * (variables + 1) + col];
        } else {
            tableau[offt] = 0;
        }
    }
    else if (col < columns - 1) {
        tableau[offt] = (row == col - variables || row == rows - 1) ? 1 : 0;
    }
    else {
        tableau[offt] = problem[row * (variables + 1) + variables];
    }
}


__kernel void setupTableauPhaseII(
            global double* problem,
            global double* tableau_p1,
            global double* tableau,
            const uint variables,
            const uint slacks_p1,
            const int objective_sign)
{
    int rows = get_global_size(0);
    int columns = get_global_size(1);
    int columns_p1 = variables + slacks_p1 + 1;

    int row = get_global_id(0);
    int col = get_global_id(1);
    int offt = row * columns + col;

    if (col < variables) {
        if (row < rows - 1) {
            tableau[offt] = tableau_p1[row * columns_p1 + col];
        } else {
            tableau[offt] =
                objective_sign * problem[row * (variables + 1) + col];
        }
    }
    else if (col < columns - 1) {
        tableau[offt] = (row == col - variables || row == rows - 1) ? 1 : 0;
    }
    else if (row == rows - 1) {
        tableau[offt] = 0;
    } else {
        tableau[offt] = tableau_p1[(row + 1) * columns_p1 - 1];
    }

    if (row == rows - 1) {
        tableau_p1[offt] = tableau[offt];
    }
}


__kernel void prepareToSolvePhaseI(
            global double* tableau,
            const int2 size,
            const int slacks)
{
    int id = get_global_id(0);
    if (id < size.y - slacks - 1 || id == size.y - 1) {
        double sum = 0;
        for (int i = 0; i < slacks; ++i) {
            sum += tableau[i * size.y + id];
        }
        tableau[(size.x - 1) * size.y + id] -= sum;
    } else {
        tableau[(size.x - 1) * size.y + id] = 0;
    }
}


__kernel void detectBasicSolutionMembers(
            global double* tableau,
            global int* indexBuf,
            const int2 size)
{
    int id = get_global_id(0);

    int row = -1;
    for (int r = 0; r < size.x - 1;++r) {
        int offt = r * size.y + id;
        if (fabs(tableau[offt] - 1) < EPSILON) {
            if (row == -1) {
                row = r;
            } else {
                row = -1;
                break;
            }
        } else if (fabs(tableau[offt]) > EPSILON) {
            row = -1;
            break;
        }
    }

    indexBuf[id] = row;
}


__kernel void zeroOutBasicSolutionMembersObjective(
            global double* tableau,
            global double* tableauOut,
            global int* indexBuf,
            const int2 size,
            const int nRows)
{
    int id = get_global_id(0);
    int offt = (size.x - 1) * size.y + id;
    
    double new_value = tableau[offt];
    for (int i = 0; i < nRows; ++i) {
        if (indexBuf[i] >= 0) {
            new_value -= tableauOut[indexBuf[i] * size.y + id] * tableau[(size.x - 1) * size.y + i];
        }
    }

    tableauOut[offt] = new_value;
}




__kernel void selectPivot(global double* tableau, global int* indexBuf,
                          const int2 tableSz, const int2 partition)
{
    const int id = get_global_id(0);

    double current = INFINITY;
    int current_idx = -1;
    int pivotColumn = -1;

    if (id < partition.y) {
        const int count = max(1, (tableSz.y - 2) / partition.y);
        const int from = id * count;
        int to = from + count;
        if (id == partition.y - 1) {
            to = tableSz.y - 2;
        }
        
        global double* row = tableau + (tableSz.x - 1) * tableSz.y;
        for (int i = from; i < to; ++i) {
            const double v = row[i];
            if (v < 0 && v < current) {
                current_idx = i;
                current = v;
            }
        }

        indexBuf[id] = current_idx;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (id == 0) {
        global double* row = tableau + (tableSz.x - 1) * tableSz.y;
        for (int i = 1; i < partition.y; ++i) {
            if (indexBuf[i] > 0) {
                const double v = row[indexBuf[i]];
                if (v < 0 && v < current) {
                    current_idx = indexBuf[i];
                    current = v;
                }
            }
        }
        indexBuf[0] = current_idx;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    pivotColumn = indexBuf[0];
    if (pivotColumn < 0) {
        if (id == 0) {
            indexBuf[1] = -1;
        }
        return;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    current = INFINITY;
    current_idx = -1;

    if (id < partition.x) {
        const int count = max(1, (tableSz.x - 1) / partition.x);
        const int from = id * count;
        int to = from + count;
        if (id == partition.x - 1) {
            to = tableSz.x - 1;
        }
 
        for (int i = from; i < to; ++i) {
            const double v = tableau[pivotColumn + i * tableSz.y];
            if (v < EPSILON) {
                continue;
            }

            const double r = tableau[(i + 1) * tableSz.y - 1] / v;

            if (r > 0 && r < current) {
                current_idx = i;
                current = r;
            }
        }

        indexBuf[id] = current_idx;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (id == 0) {
        for (int i = 1; i < partition.x; ++i) {
            if (indexBuf[i] > 0) {
                const double a = tableau[pivotColumn + indexBuf[i] * tableSz.y];
                const double b = tableau[(indexBuf[i] + 1) * tableSz.y - 1];
                const double r = b / a;
                if (r > 0 && r < current) {
                    current_idx = indexBuf[i];
                    current = r;
                }
            }
        }
        indexBuf[0] = pivotColumn;
        indexBuf[1] = current_idx;
    }
}


__kernel void clearWithPivot(
            global double* tableau,
            global double* tableauOut,
            const uint2 pivot)
{
    const uint2 pos = {get_global_id(0), get_global_id(1)};
    const int stride = get_global_size(0);

    double p = tableau[pivot.y * stride + pivot.x];
    if (pos.y == pivot.y) {
        // the pivot row!
        uint offt = pos.y * stride + pos.x;
        tableauOut[offt] = tableau[offt] / p;
    } else {
        double v = tableau[pos.y * stride + pivot.x];

        uint offt = pos.y * stride + pos.x;
        if (pos.x == pivot.x) {
            tableauOut[offt] = 0;
        } else {
            tableauOut[offt] = tableau[offt] - (v / p) * tableau[pivot.y * stride + pos.x];
        }
    }
}

__kernel void buildSolutionTable(
            global double* tableau,
            global double* solution)
{
    const uint2 pos = {get_global_id(0), get_global_id(1)};
    const uint stride = get_global_size(1);
    
    uint offt = pos.x * stride + pos.y;
    double v0 = tableau[offt];
    if (fabs(v0) > EPSILON) {
        solution[offt] = tableau[(pos.x + 1) * stride - 1] / v0;
    } else {
        solution[offt] = 0;
    }
}

__kernel void extractSolution(
            global double* solution,
            const uint stride,
            const uint maxIndex) {
    const uint this_pos  =  get_global_id(0)                       * stride + get_global_id(1);
    const uint other_pos = (get_global_id(0) + get_global_size(0)) * stride + get_global_id(1);
    
    if (other_pos < maxIndex && !isinf(solution[this_pos])) {
        if (solution[this_pos] == 0) {
            solution[this_pos] = solution[other_pos];
        } else if (solution[other_pos] != 0) {
            solution[this_pos] = -INFINITY;
        }
        solution[other_pos] = 0;
    }
}

__kernel void fixInfinities(global double* solution) {
    uint id = get_global_id(0);
    
    if (isinf(solution[id])) {
        solution[id] = 0;
    }
}

