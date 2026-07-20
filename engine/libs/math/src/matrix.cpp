#include <slm/math/matrix.hpp>

#include <stdexcept>

namespace slm::math {

Matrix::Matrix(const std::size_t rows,
               const std::size_t columns,
               const float initial_value)
    : rows_(rows), columns_(columns), values_(rows * columns, initial_value) {}

Matrix::Matrix(const std::size_t rows,
               const std::size_t columns,
               const std::initializer_list<float> values)
    : rows_(rows), columns_(columns), values_(values) {
    if (values_.size() != rows_ * columns_) {
        throw std::invalid_argument("matrix value count does not match its shape");
    }
}

Matrix::Matrix(const std::initializer_list<std::initializer_list<float>> rows)
    : rows_(rows.size()), columns_(rows.size() == 0 ? 0 : rows.begin()->size()) {
    values_.reserve(rows_ * columns_);
    for (const std::initializer_list<float> row_values : rows) {
        if (row_values.size() != columns_) {
            throw std::invalid_argument("every matrix row must have equal length");
        }
        values_.insert(values_.end(), row_values.begin(), row_values.end());
    }
}

std::size_t Matrix::rows() const noexcept {
    return rows_;
}

std::size_t Matrix::columns() const noexcept {
    return columns_;
}

std::size_t Matrix::size() const noexcept {
    return values_.size();
}

bool Matrix::empty() const noexcept {
    return values_.empty();
}

float& Matrix::operator()(const std::size_t row, const std::size_t column) {
    return values_.at(index(row, column));
}

const float& Matrix::operator()(const std::size_t row,
                                const std::size_t column) const {
    return values_.at(index(row, column));
}

float* Matrix::data() noexcept {
    return values_.data();
}

const float* Matrix::data() const noexcept {
    return values_.data();
}

std::vector<float> Matrix::row(const std::size_t row_index) const {
    if (row_index >= rows_) {
        throw std::out_of_range("matrix row is outside its shape");
    }

    std::vector<float> result(columns_);
    for (std::size_t column_index = 0; column_index < columns_; ++column_index) {
        result[column_index] = (*this)(row_index, column_index);
    }
    return result;
}

std::vector<float> Matrix::column(const std::size_t column_index) const {
    if (column_index >= columns_) {
        throw std::out_of_range("matrix column is outside its shape");
    }

    std::vector<float> result(rows_);
    for (std::size_t row_index = 0; row_index < rows_; ++row_index) {
        result[row_index] = (*this)(row_index, column_index);
    }
    return result;
}

std::size_t Matrix::index(const std::size_t row,
                          const std::size_t column) const {
    if (row >= rows_ || column >= columns_) {
        throw std::out_of_range("matrix index is outside its shape");
    }
    return row * columns_ + column;
}

}  // namespace slm::math
