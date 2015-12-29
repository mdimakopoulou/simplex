
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

__kernel void selectNegatives(
            global double* tableau,
            global int* indexBuf,
            const uint offsetToLastRow) {
    uint id = get_global_id(0);
    indexBuf[id] = (tableau[offsetToLastRow + id] < 0) ? id : (-1);
}

__kernel void findMin(
            global double* tableau,
            global int* indexBuf,
            const uint otherOffset,
            const uint maxIndex,
            const uint offset) {
    uint id = get_global_id(0);
    uint other = id + otherOffset;

    if (other < maxIndex) {
        if (indexBuf[id] < 0) {
            indexBuf[id] = indexBuf[other];
        } else if (indexBuf[other] >= 0) {
            if (tableau[offset + indexBuf[other]] < tableau[offset + indexBuf[id]]) {
                indexBuf[id] = indexBuf[other];
            }
        }
    }
}

__kernel void selectMinNegative(
            global double* tableau,
            global int* indexBuf,
            const uint tOffset,
            const uint tCount) {
    double current = INFINITY;
    int current_idx = -1;
    uint id = get_global_id(0);
    uint count = max(1u, tCount / get_global_size(0));
    uint end = count;
    if (id == get_global_size(0) - 1) {
        end += tCount % get_global_size(0);
    }
    
    uint begin = id * count;
    for (uint i = 0, offt = tOffset + begin; i < end; ++i, ++offt) {
        double v = tableau[offt];
        if (v < 0 && v < current) {
            current_idx = i + begin;
            current = v;
        }
    }

    indexBuf[id] = current_idx;

    if (get_global_size(0) > 1) {
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (id == 0) {
            for (uint i = 1; i < get_global_size(0); ++i) {
                if (indexBuf[i] > 0) {
                    double v = tableau[tOffset + indexBuf[i]];
                    if (v < 0 && v < current) {
                        current_idx = indexBuf[i];
                        current = v;
                    }
                }
            }
            indexBuf[0] = current_idx;
        }
    }
}


__kernel void selectMinRatio(
            global double* tableau,
            global int* indexBuf,
            const uint tOffset,
            const uint tCount,
            const uint tStride,
            const uint nOffset) {
    double current = INFINITY;
    int current_idx = -1;

    uint id = get_global_id(0);
    uint count = max(1u, tCount / get_global_size(0));
    uint end = count;
    if (id == get_global_size(0) - 1) {
        end += tCount % get_global_size(0);
    }
    
    uint begin = id * count;
    uint base = tOffset + begin * tStride;
    uint nbase = nOffset + begin * tStride;
    for (uint i = 0, offt = 0; i < end; ++i, offt += tStride) {
        double v = tableau[base + offt];
        if (v < EPSILON) {
            continue;
        }

        double n = tableau[nbase + offt];
        v = n / v;

        if (v > 0 && v < current) {
            current_idx = i + begin;
            current = v;
        }
    }

    indexBuf[id] = current_idx;

    if (get_global_size(0) > 1) {
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (id == 0) {
            for (uint i = 1; i < get_global_size(0); ++i) {
                if (indexBuf[i] > 0) {
                    double a = tableau[tOffset + indexBuf[i] * tStride];
                    double b = tableau[nOffset + indexBuf[i] * tStride];
                    double v = b / a;
                    if (v > 0 && v < current) {
                        current_idx = indexBuf[i];
                        current = v;
                    }
                }
            }
            indexBuf[0] = current_idx;
        }
    }
}




double ratioAt(global double* tableau, uint2 index) {
    double a = tableau[index.x];
    if (a <= EPSILON) return INFINITY;

    double b = tableau[index.y];
    double r = b / a;

    return r <= EPSILON ? INFINITY : r;
}



__kernel void copyPivotColumnRatios(
            global double* tableau,
            global int* indexBuf,
            const uint2 offset,
            const uint stride,
            const uint otherOffset,
            const uint maxIndex) {
    uint id = get_global_id(0);
    uint other = id + otherOffset;
    
    double r = ratioAt(tableau, offset + id * stride);
    if (other >= maxIndex) {
        indexBuf[id] = isinf(r) ? -1 : id;
    } else {
        double ro = ratioAt(tableau, offset + other * stride);
        if (isinf(r)) {
            indexBuf[id] = isinf(ro) ? -1 : other;
        } else {
            if (isinf(ro) || ro > r) {
                indexBuf[id] = id;
            } else {
                indexBuf[id] = other;
            }
        }
    }
}

__kernel void findMinRatio(
            global double* tableau,
            global int* indexBuf,
            const uint2 offset,
            const uint stride,
            const uint otherOffset,
            const uint maxIndex) {
    uint id = get_global_id(0);
    uint other = id + otherOffset;

    if (other < maxIndex) {
        if (indexBuf[id] < 0) {
            indexBuf[id] = indexBuf[other];
        } else {
            if (indexBuf[other] >= 0) {
                double r = ratioAt(tableau, offset + indexBuf[id] * stride);
                if (isinf(r)) {
                    indexBuf[id] = indexBuf[other];
                } else {
                    double ro = ratioAt(tableau, offset + indexBuf[other] * stride);
                    if (!isinf(ro) && ro < r) {
                        indexBuf[id] = indexBuf[other];
                    }
                }
            }
        }
    }
}

__kernel void clearWithPivot(
            global double* tableau,
            global double* tableauOut,
            const uint2 pivot,
            const uint stride) {
    const uint2 pos = {get_global_id(0), get_global_id(1)};

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
            global double* solution,
            const uint stride) {
    const uint2 pos = {get_global_id(0), get_global_id(1)};
    
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

