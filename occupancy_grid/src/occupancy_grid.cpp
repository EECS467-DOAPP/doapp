#include <occupancy_grid/occupancy_grid.hpp>

#include <cstring>

namespace doapp
{

OccupancyGrid::OccupancyGrid(double size, double granularity) : size_(size),
                                                                granularity_(granularity_),
                                                                cells_per_edge_(static_cast<size_t>(size / granularity)),
                                                                num_cells_(cells_per_edge_ * cells_per_edge_ * cells_per_edge_),
                                                                bounds_(size / 2)
{
    // cudaMalloc(grid_, num_cells * sizeof(int8_t));
    grid_ = new int8_t[num_cells_];
    memset(grid_, 0, sizeof(int8_t) * num_cells_);
}

OccupancyGrid::~OccupancyGrid()
{
    delete[] grid_;
}

int8_t OccupancyGrid::get(size_t x, size_t y, size_t z)
{
    return grid_[x + (cells_per_edge_ * y) + (cells_per_edge_ * cells_per_edge_ * z)];
}

int8_t OccupancyGrid::get(double x, double y, double z)
{
    const double x_shifted = x + size_ / 2.0;
    const double y_shifted = y + size_ / 2.0;
    const double z_shifted = z + size_ / 2.0;

    const size_t x_discrete = static_cast<size_t>(x_shifted / granularity_);
    const size_t y_discrete = static_cast<size_t>(y_shifted / granularity_);
    const size_t z_discrete = static_cast<size_t>(z_shifted / granularity_);

    return get(x_discrete, y_discrete, z_discrete);
}

void OccupancyGrid::set(size_t x, size_t y, size_t z, int8_t val)
{
    grid_[x + (cells_per_edge_ * y) + (cells_per_edge_ * cells_per_edge_ * z)] = val;
}

void OccupancyGrid::set(double x, double y, double z, int8_t val)
{
    const double x_shifted = x + size_ / 2.0;
    const double y_shifted = y + size_ / 2.0;
    const double z_shifted = z + size_ / 2.0;

    const size_t x_discrete = static_cast<size_t>(x_shifted / granularity_);
    const size_t y_discrete = static_cast<size_t>(y_shifted / granularity_);
    const size_t z_discrete = static_cast<size_t>(z_shifted / granularity_);

    set(x_discrete, y_discrete, z_discrete, val);
}

const int8_t &OccupancyGrid::operator()(size_t x, size_t y, size_t z) const
{
    return grid_[x + (cells_per_edge_ * y) + (cells_per_edge_ * cells_per_edge_ * z)];
}

const int8_t &OccupancyGrid::operator()(double x, double y, double z) const
{
    const double x_shifted = x + size_ / 2.0;
    const double y_shifted = y + size_ / 2.0;
    const double z_shifted = z + size_ / 2.0;

    const size_t x_discrete = static_cast<size_t>(x_shifted / granularity_);
    const size_t y_discrete = static_cast<size_t>(y_shifted / granularity_);
    const size_t z_discrete = static_cast<size_t>(z_shifted / granularity_);

    return operator()(x_shifted,
                      y_shifted,
                      z_shifted);
}

int8_t &OccupancyGrid::operator()(size_t x, size_t y, size_t z)
{
    return grid_[x + (cells_per_edge_ * y) + (cells_per_edge_ * cells_per_edge_ * z)];
}

int8_t &OccupancyGrid::operator()(double x, double y, double z)
{
    const double x_shifted = x + size_ / 2.0;
    const double y_shifted = y + size_ / 2.0;
    const double z_shifted = z + size_ / 2.0;

    const size_t x_discrete = static_cast<size_t>(x_shifted / granularity_);
    const size_t y_discrete = static_cast<size_t>(y_shifted / granularity_);
    const size_t z_discrete = static_cast<size_t>(z_shifted / granularity_);

    return operator()(x_shifted,
                      y_shifted,
                      z_shifted);
}

void OccupancyGrid::serialize(occupancy_grid::OccupancyGrid &map) const
{
    map.size = size_;
    map.granularity = granularity_;
    map.grid = std::vector<int8_t>(grid_, grid_ + num_cells_);
}

void OccupancyGrid::deserialize(const occupancy_grid::OccupancyGrid &map)
{
    if (map.size == size_ && map.granularity == granularity_)
    {
        std::copy(map.grid.begin(), map.grid.end(), grid_);
    }
}

} // namespace doapp
