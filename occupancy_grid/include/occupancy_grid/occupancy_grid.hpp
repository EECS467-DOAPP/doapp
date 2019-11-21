#include <cstdint>
#include <cstdlib>
#include <vector>

#include <occupancy_grid/OccupancyGrid.h>

namespace doapp
{

class OccupancyGrid
{
public:
    // Creates a cube occupancy grid.
    // size: length of an edge of a cube (meters)
    // granularity: length of an edge of a grid cell (meters)
    OccupancyGrid(double size, double granularity);

    ~OccupancyGrid();

    int8_t get(size_t x, size_t y, size_t z);
    int8_t get(double x, double y, double z);

    void set(size_t x, size_t y, size_t z, int8_t val);
    void set(double x, double y, double z, int8_t val);

    void serialize(occupancy_grid::OccupancyGrid &map) const;
    void deserialize(const occupancy_grid::OccupancyGrid &map);

private:
    const double size_;
    const double granularity_;

    const size_t cells_per_edge_;
    const size_t num_cells_;

    int8_t *grid_;

public:
    const double bounds_;
};

} // namespace doapp
