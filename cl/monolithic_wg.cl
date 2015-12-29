
#define EPSILON     1e-20


__kernel void setupTableau(
            global double* problem,
            global double* tableau,
            const uint variables)
{
    int t_stride = get_global_size(1);
    int t_offt = get_global_id(0) * t_stride + get_global_id(1);

    if (get_global_id(1) < variables - 1) {
        int sign = get_global_id(0) == get_global_size(1) - variables - 1 ? -1 : 1;
        tableau[t_offt] = sign * problem[get_global_id(0) * variables + get_global_id(1)];
    }
    else if (get_global_id(1) == get_global_size(1) - 1) {
        tableau[t_offt] = problem[get_global_id(0) * variables + variables - 1];
    }
    else if (get_global_id(0) - 1 == get_global_id(1) - variables) {
        tableau[t_offt] = 1;
    }
    else {
        tableau[t_offt] = 0;
    }
}


#define STATUS_OK           0
#define STATUS_ITR_BRK      1
#define STATUS_NO_OPTIMAL   2


int indexOfMinNegative(global double* row, global int* indexBuf, const int columns, const int partition) {
    barrier(CLK_GLOBAL_MEM_FENCE);

    const int id = get_global_id(0);
    double current = INFINITY;
    int current_idx = -1;

    if (id < partition) {
        const int count = max(1, (columns - 2) / partition);
        const int from = id * count;
        int to = from + count;
        if (id == partition - 1) {
            to = columns - 2;
        }
        
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
        for (int i = 1; i < partition; ++i) {
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

    return indexBuf[0];
}


int indexOfMinRatio(global double* buf, global int* indexBuf, const int column, const int columns, const int rows, const int partition) {
    barrier(CLK_GLOBAL_MEM_FENCE);

    const int id = get_global_id(0);
    double current = INFINITY;
    int current_idx = -1;

    if (id < partition) {
        const int count = max(1, (rows - 1) / partition);
        const int from = id * count;
        int to = from + count;
        if (id == partition - 1) {
            to = rows - 1;
        }
 
        for (int i = from; i < to; ++i) {
            const double v = buf[column + i * columns];
            if (v < EPSILON) {
                continue;
            }

            const double r = buf[(i + 1) * columns - 1] / v;

            if (r > 0 && r < current) {
                current_idx = i;
                current = r;
            }
        }

        indexBuf[id] = current_idx;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (id == 0) {
        for (int i = 1; i < partition; ++i) {
            if (indexBuf[i] > 0) {
                const double a = buf[column + indexBuf[i] * columns];
                const double b = buf[(indexBuf[i] + 1) * columns - 1];
                const double r = b / a;
                if (r > 0 && r < current) {
                    current_idx = indexBuf[i];
                    current = r;
                }
            }
        }
        indexBuf[0] = current_idx;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    return indexBuf[0];
}


void clearWithPivot(global double* read_buf, global double* write_buf, const int2 pivot, const int2 pos, const int columns) {
    const int offt = pos.x * columns + pos.y;

    const double p = read_buf[pivot.x * columns + pivot.y];

    if (pos.x == pivot.x) {
        write_buf[offt] = read_buf[offt] / p;
        return;
    }

    const double v = read_buf[pos.x * columns + pivot.y];
    if (pos.y == pivot.y) {
        write_buf[offt] = 0;
    } else {
        write_buf[offt] = read_buf[offt] - (v / p) * read_buf[pivot.x * columns + pos.y];
    }
}


int getSingleNonZeroRow(global double* buf, const int rows, const int columns, const int column) {
    int nonZeroRow = -1;
    
    for (int r = 0; r < rows; ++r) {
        const double v = buf[r * columns + column];
        if (fabs(v) > EPSILON) {
            if (nonZeroRow < 0) {
                nonZeroRow = r;
            } else {
                return -1;
            }
        }
    }

    return nonZeroRow;
}


void extractSolution(global double* read_buf, global double* write_buf, const int rows, const int columns, const int col, const int variables) {
    const int tabSz = rows * columns;

    if (col == variables) {
        const double prev_to_last = read_buf[tabSz - 2];
        write_buf[variables] = fabs(prev_to_last) > EPSILON ? read_buf[tabSz - 1] / prev_to_last : 0;
    } else {
        const int row = getSingleNonZeroRow(read_buf, rows, columns, col);
        if (row < 0) {
            write_buf[col] = 0;
        } else {
            const double v = read_buf[row * columns + col];
            const double n = read_buf[(row + 1) * columns - 1];
            write_buf[col] = n / v;
        }
    }
}


__kernel void monolithicSimplex(
            global double* buf0,
            global double* buf1,
            global int* indexBuf,
            const int2 tableSize, // rows, columns
            const int2 partition, // rows, column threads to use for scanning
            const int maxSteps,
            const int variables)
{
    const int threads_no = get_global_size(0);
    const int table_size = tableSize.x * tableSize.y;
    const int id = get_global_id(0);

    int status = STATUS_ITR_BRK;
    int steps = 0;
    int2 pivot = { 0, 0 };

    global double* read_buf = buf0;
    global double* write_buf = buf1;

    const int last_row_offt = (tableSize.x - 1) * tableSize.y;

    while (steps < maxSteps) {
        read_buf  = steps%2 == 0 ? buf0 : buf1;
        write_buf = steps%2 == 0 ? buf1 : buf0;

        global double* last_row = read_buf + last_row_offt;

        pivot.y = indexOfMinNegative(last_row, indexBuf, tableSize.y, partition.x);
        if (pivot.y < 0) {
            pivot.x = -1;
            status = STATUS_OK;
            break;
        }

        pivot.x = indexOfMinRatio(read_buf, indexBuf, pivot.y, tableSize.y, tableSize.x, partition.y);
        if (pivot.x < 0) {
            status = STATUS_NO_OPTIMAL;
            break;
        }

        for (int n = id; n < table_size; n += threads_no) {
            int2 pos = { n / tableSize.y, n % tableSize.y };
            clearWithPivot(read_buf, write_buf, pivot, pos, tableSize.y);
        }
        ++steps;
    }

    if (status == STATUS_OK) {
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int n = id; n <= variables; n += threads_no) {
            extractSolution(read_buf, write_buf, tableSize.x, tableSize.y, n, variables);
        }
    }

    if (id == 0) {
        indexBuf[0] = steps;
        indexBuf[1] = status;
        indexBuf[2] = pivot.x;
        indexBuf[3] = pivot.y;
    }
}

