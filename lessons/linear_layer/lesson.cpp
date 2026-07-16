#include <slm/engine/controls.hpp>
#include <slm/engine/lesson.hpp>
#include <slm/engine/show.hpp>
#include <slm/math/matrix.hpp>
#include <slm/math/vector.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using slm::math::Matrix;
using slm::math::Vector;

// This is the concept being learned. Once it is understood and needed by later
// lessons, it can be promoted unchanged into libs/math/linear.hpp.
Matrix linear_forward(const Matrix& inputs,
                      const Matrix& weights,
                      const Vector& biases) {
    if (inputs.columns() != weights.columns()) {
        throw std::invalid_argument("input and weight feature counts must match");
    }
    if (biases.size() != weights.rows()) {
        throw std::invalid_argument("each neuron requires one bias");
    }

    Matrix outputs(inputs.rows(), weights.rows());

    // Y = X * W^T + b
    for (std::size_t sample = 0; sample < inputs.rows(); ++sample) {
        for (std::size_t neuron = 0; neuron < weights.rows(); ++neuron) {
            float weighted_sum = 0.0F;

            // Dot product between one sample and one neuron's weights.
            for (std::size_t feature = 0; feature < inputs.columns(); ++feature) {
                weighted_sum +=
                    inputs(sample, feature) * weights(neuron, feature);
            }

            outputs(sample, neuron) = weighted_sum + biases[neuron];
        }
    }

    return outputs;
}

struct LinearLayerState {
    Matrix inputs{
        {1.0F, 2.0F, 3.0F},
        {1.5F, 2.5F, 3.5F},
        {2.0F, 1.0F, 4.0F},
        {0.5F, 3.0F, 2.0F},
        {3.0F, 4.0F, 5.0F},
    };
    Matrix weights{
        {0.1F, 0.2F, 0.3F},
        {0.4F, 0.5F, 0.6F},
    };
    Vector biases{0.5F, 1.0F};
    int selected_neuron = 0;
    bool use_output_as_z = true;
    bool connect_points = false;
};

}  // namespace

SLM_LESSON(linear_layer, "Linear Layer", "Linear Algebra", 30) {
    // Static means this ordinary C++ state keeps its values between frames.
    static LinearLayerState state;

    slm::controls::note(
        "Change W and b, then watch the forward pass update immediately.");
    slm::controls::matrix("Weights W", state.weights, -2.0F, 2.0F, "%.2f");
    slm::controls::vector("Bias b", state.biases, -5.0F, 5.0F, "%.2f");
    slm::controls::choice(
        "Selected neuron", state.selected_neuron, {"Neuron 0", "Neuron 1"});
    slm::controls::toggle("Use neuron output as Z", state.use_output_as_z);
    slm::controls::toggle("Connect sample points", state.connect_points);
    if (slm::controls::button("Reset lesson")) {
        state = LinearLayerState{};
    }

    // Educational datasets are small, so recomputing every frame keeps the
    // relationship between controls, math, and output straightforward.
    const Matrix outputs =
        linear_forward(state.inputs, state.weights, state.biases);

    slm::show::text(
        "Equation",
        "Y = X * W^T + b\n\n"
        "For every sample and neuron:\n"
        "1. Multiply matching input features and weights.\n"
        "2. Add those products to form a dot product.\n"
        "3. Add the neuron's bias.");
    slm::show::matrix("Inputs X", state.inputs, "Rows are samples; columns are features.");
    slm::show::matrix("Weights W", state.weights, "Each row belongs to one neuron.");
    slm::show::matrix("Outputs Y", outputs, "Each column is one neuron's output.");

    const Vector z_values = state.use_output_as_z
                                ? outputs.column(static_cast<std::size_t>(
                                      state.selected_neuron))
                                : state.inputs.column(2);
    slm::show::plot3d("Mapping",
                      state.inputs.column(0),
                      state.inputs.column(1),
                      z_values,
                      {"Input x0", "Input x1", "Selected Z"},
                      state.connect_points);
}
