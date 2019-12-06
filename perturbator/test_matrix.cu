#include "matrix.cu"

#include <iostream>

int main() {
    doapp::Matrix<float, 3, 3> A(3, 3);

    A[0][0] = 1.0f;
    A[0][1] = 2.0f;
    A[0][2] = 3.0f;

    A[1][0] = 4.0f;
    A[1][1] = 5.0f;
    A[1][2] = 6.0f;

    A[2][0] = 7.0f;
    A[2][1] = 8.0f;
    A[2][2] = 9.0f;

    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        const auto this_row = A[i];

        if (A.num_cols() > 0) {
            std::cout << this_row[0];

            for (std::size_t j = 1; j < A.num_cols(); ++j) {
                std::cout << ", " << this_row[j];
            }
        }

        std::cout << '\n';
    }
}
