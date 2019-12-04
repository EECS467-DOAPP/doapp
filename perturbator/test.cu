#include "random.h"
#include "vector.h"

#include <iostream>

int main() {
  doapp::Vector<doapp::Pcg32, doapp::Dynamic> generators =
      doapp::Pcg32::from_randomness(64);

  for (std::size_t i = 0; i < generators.size(); ++i) {
    std::cout << "generators[" << i << "].randf() = " << generators[i].randf()
              << '\n';
  }
}
