#ifndef _SIMPLEX_MATRIX_H
#define _SIMPLEX_MATRIX_H

#include "main.h"


class Matrix {
public:
    Matrix(uint32_t rows, uint32_t columns);
    ~Matrix();

    static Matrix* loadCSV(const char* filename, char delim = ',');

    static Matrix* makeRandom(long seed, uint32_t rows, uint32_t columns,
                              double absMax, bool integerValues,
                              float zeroProbability,
                              bool allowZeroAtLastRow,
                              bool allowNegAtLastColumn);

    uint32_t size() const { return rows() * columns(); }
    void print() const;
    void toCSV(const char* filename, char delim = ',') const;

    uint32_t rows() const { return rows_; }
    uint32_t columns() const { return columns_; }

    double at(int row, int column) const { return elements_[row * columns_ + column]; }
    double& at(int row, int column) { return elements_[row * columns_ + column]; }

    // copy: this[box @ row0,column0] = source[box @ row1,column1] 
    // nRows, nColumns: when 0 (zero) the maximum possible box will be copied.
    void copy(const Matrix& source,
              uint32_t row0 = 0, uint32_t column0 = 0,
              uint32_t row1 = 0, uint32_t column1 = 0,
              uint32_t nRows = 0, uint32_t nColumns = 0);

    void setIdentityAt(uint32_t width, uint32_t row0 = 0, uint32_t column0 = 0);

    void multiplyRow(uint32_t row, double factor);
    void maddRow(uint32_t sourceRow, uint32_t destinationRow, double factor);

    double* rawptr() { return elements_; }

private:
    const uint32_t rows_;
    const uint32_t columns_;
    double* elements_;
};

#endif
