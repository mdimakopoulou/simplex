#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cmath>
#include <random>

#include "matrix.h"

using namespace std;


Matrix::Matrix(uint32_t rows, uint32_t columns)
    : rows_(rows), columns_(columns), elements_(new double[rows * columns])
{
    memset(elements_, 0, sizeof(double) * rows * columns);
}

Matrix::~Matrix() {
    if (elements_)
        delete [] elements_;
}

void Matrix::copy(const Matrix& source,
                  uint32_t row0, uint32_t column0,
                  uint32_t row1, uint32_t column1,
                  uint32_t nRows, uint32_t nColumns) {
    if (nRows == 0) {
        nRows = min(rows_ - row0, source.rows_ - row1);
    }
    if (nColumns == 0) {
        nColumns = min(columns_ - column0, source.columns_ - column1);
    }

    for (uint32_t r = row0, rEnd = row0 + nRows; r < rEnd; ++r) {
        memcpy(elements_ + r * columns_ + column0,
               source.elements_ + (r - row0 + row1) * source.columns_ + column1,
               sizeof(double) * nColumns);
    }
}

void Matrix::setIdentityAt(uint32_t width, uint32_t row0, uint32_t column0) {
    for (uint32_t r = row0, rEnd = row0 + width; r < rEnd; ++r) {
        memset(elements_ + r * columns_ + column0, 0, sizeof(double) * width);
        elements_[ r * columns_ + column0 + r - row0] = 1;
    }
}

void Matrix::multiplyRow(uint32_t row, double factor) {
    uint32_t rowOffset = row * columns_;
    for (uint32_t c = 0; c < columns_; ++c) {
        elements_[rowOffset + c] *= factor;
    }
}

void Matrix::maddRow(uint32_t sourceRow, uint32_t destinationRow, double factor) {
    uint32_t srcOfft = sourceRow * columns_;
    uint32_t dstOfft = destinationRow * columns_;
    for (uint32_t c = 0; c < columns_; ++c) {
        elements_[dstOfft + c] += factor * elements_[srcOfft + c];
    }
}

void Matrix::print() const {
    const size_t count = rows_ * columns_;

    // pre-process: detect size of string formatted numbers to align the table
    int digits = 0, decimals = 2;
    int decimalFactor = 100;
    bool hasNegatives = false;

    for (size_t i = 0; i < count; ++i) {
        hasNegatives |= elements_[i] < 0;

        double n = abs(elements_[i]);
        digits = max(digits, (int)ceil(log10(n)));
        
        if(n < 1 && n > 0) {
            while (decimalFactor * n < 1 && decimalFactor < 6) {
                ++decimals;
                decimalFactor *= 10;
            }
        }
    }

    // actual printing: use field width for alignment & cache precision so as to restore afterwards
    const int width = digits + decimals + (hasNegatives ? 4 : 2);
    int precision = cout.precision();
    cout.precision(decimals);
    for (size_t i = 0; i < count; ++i) {
        if (i && i%columns_ == 0)
            cout << endl;
        cout << setw(width) << elements_[i];
    }
    cout.precision(precision);
    cout << setw(1);

    cout << endl;
}

void Matrix::toCSV(const char* filename, char delim) const {
    ofstream output(filename);

    for (size_t i = 0, count = rows_ * columns_; i < count; ++i) {
        if (i) {
            if (i%columns_ == 0) {
                output << endl;
            } else {
                output << delim;
            }
        }
        output << fabs(elements_[i]) * (elements_[i] < 0 ? -1 : 1);
    }
    output << endl;

    output.close();
}

Matrix* Matrix::loadCSV(const char* filename, char delim) {
    char buffer[1024];
    int lineCount = 0;
    bool fail = false;
    ifstream input(filename);

    vector<vector<double>> temp_table;
    unsigned columns = 0; // used to ensure all rows have the same column count

    while (!input.eof()) {
        input.getline(buffer, sizeof(buffer));
        if (input.fail()) {
            if (!input.eof()) {
                cerr << "Failed reading input: " << filename << " after line: " << lineCount << endl;
                fail = true;
            }
            break;
        }
        ++lineCount;

        istringstream line(buffer);
        string number_str;
        vector<double> numbers;
        while (getline(line, number_str, delim)) {
            size_t idx = 0;
            double n = stod(number_str, &idx);
            if (idx != number_str.length()) {
                cerr << "Invalid number [" << number_str << "] @ cell {" << lineCount << "," << (numbers.size() + 1) << "}\n";
                fail = true;
                break;
            }
            numbers.push_back(n);
        }
        if (columns == 0) {
            columns = numbers.size();
        } else if(columns != numbers.size()) {
            cerr << "Invalid column count @ line:" << numbers.size() << " , expected:" << columns << endl;
            fail = true;
            break;
        }

        temp_table.push_back(numbers);
    }
    input.close();

    Matrix* result = nullptr;

    // all errors are fatal...
    if (!fail) {
        result = new Matrix(temp_table.size(), columns);
        int n = 0;
        for (auto row: temp_table) {
            for (double value: row) {
                result->elements_[n++] = value;
            }
        }
    }

    return result;
}

Matrix* Matrix::makeRandom(long seed, uint32_t rows, uint32_t columns,
                           double absMax, bool integerValues,
                           float zeroProbability, bool allowZeroAtLastRow, bool allowNegAtLastColumn) {
    Matrix* result = new Matrix(rows, columns);

    mt19937 prng;
    prng.seed(seed);
    double norm = prng.max();

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < columns; ++c) {
            bool canBeZero = allowZeroAtLastRow  || r != rows - 1;
            if (canBeZero) {
                canBeZero = prng() / norm < zeroProbability;
            }

            double v = 0;
            while (!canBeZero && v == 0) {
                if (allowNegAtLastColumn || c != columns - 1) {
                    v = 2 * (prng() / norm) - 1;
                    v *= absMax;
                } else {
                    v = absMax * prng() / norm;
                }
                if (integerValues) {
                    double s = v < 0 ? -1 : 1;
                    v = s * floor(abs(v));
                }
            }
            result->at(r, c) = v;
        }
    }

    return result;
}

