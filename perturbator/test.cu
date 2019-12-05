#include "random.cuh"

#include <iomanip>
#include <iostream>

int main() {
  doapp::Pcg32 gen;

  for (int i = 0; i < 32; ++i) {
    std::cout << "gen.rand11() = " << std::hexfloat << gen.rand11() << '\n';
  }
}
