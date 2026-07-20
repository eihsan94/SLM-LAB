#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace slm::math {

// A small row-major matrix designed to make tensor shapes and indexing visible.
// It deliberately avoids external linear-algebra libraries for this curriculum.
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t columns, float initial_value = 0.0F);
    Matrix(std::size_t rows,
           std::size_t columns,
           std::initializer_list<float> values);
    Matrix(std::initializer_list<std::initializer_list<float>> rows);

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t columns() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    float& operator()(std::size_t row, std::size_t column);
    const float& operator()(std::size_t row, std::size_t column) const;

    float* data() noexcept;
    const float* data() const noexcept;

    [[nodiscard]] std::vector<float> row(std::size_t row_index) const;
    [[nodiscard]] std::vector<float> column(std::size_t column_index) const;

private:
    [[nodiscard]] std::size_t index(std::size_t row, std::size_t column) const;

    std::size_t rows_ = 0;
    std::size_t columns_ = 0;
    std::vector<float> values_;
};

}  // namespace slm::math
