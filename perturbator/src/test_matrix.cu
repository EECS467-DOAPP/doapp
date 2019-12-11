#include "matrix.cuh"

#include <iostream>
#include <cassert>

int main() {
    doapp::Matrix<int, 2, 2> A(2, 2);
    A[0][0] = 1;
    A[0][1] = 2;
    A[1][0] = 3;
    A[1][1] = 4;

    doapp::Matrix<int, 2, 2> B(2, 2);
    B[0][0] = 5;
    B[0][1] = 6;
    B[1][0] = 7;
    B[1][1] = 8;

    const doapp::Matrix<int, 2, 2> C = A * B;

    assert(C[0][0] == 19);
    assert(C[0][1] == 22);
    assert(C[1][0] == 43);
    assert(C[1][1] == 50);

    for (std::size_t i = 0; i < C.num_rows(); ++i) {
        const auto this_row = C[i];

        if (C.num_cols() > 0) {
            std::cout << this_row[0];

            for (std::size_t j = 1; j < C.num_cols(); ++j) {
                std::cout << ", " << this_row[j];
            }
        }

        std::cout << '\n';
    }
}
