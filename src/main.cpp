#include "neural_network.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "dataset_csv_loader.h"
#include <iostream>
#include <algorithm>
#include <string>
#include <array>

using namespace utec::neural_network;
using namespace dataset;

int main() {
    // Cargar datos
    auto [X_full, Y_full] = load_csv_dataset("dataset3x3.csv");

    // Conteo de clases (X=0, O=1, Nada=2)
    std::array<int, 3> conteo_clases = {0, 0, 0};
    for (size_t i = 0; i < Y_full.shape()[0]; ++i) {
        int clase = std::distance(&Y_full(i, 0), std::max_element(&Y_full(i, 0), &Y_full(i, 0) + 3));
        ++conteo_clases[clase];
    }
    std::cout << "Distribucion de clases: X=" << conteo_clases[0]
              << ", O=" << conteo_clases[1]
              << ", Nada=" << conteo_clases[2] << "\n";

    // Separar entrenamiento y prueba (80/20)
    size_t total = X_full.shape()[0];
    size_t n_train = static_cast<size_t>(total * 0.8);
    std::array<size_t, 2> shape_x_train = {n_train, 9};
    std::array<size_t, 2> shape_x_test = {total - n_train, 9};
    std::array<size_t, 2> shape_y_train = {n_train, 3};
    std::array<size_t, 2> shape_y_test = {total - n_train, 3};

    Tensor<float, 2> X_train(shape_x_train), Y_train(shape_y_train);
    Tensor<float, 2> X_test(shape_x_test), Y_test(shape_y_test);

    for (size_t i = 0; i < total; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (i < n_train) X_train(i, j) = X_full(i, j);
            else             X_test(i - n_train, j) = X_full(i, j);
        }
        for (int j = 0; j < 3; ++j) {
            if (i < n_train) Y_train(i, j) = Y_full(i, j);
            else             Y_test(i - n_train, j) = Y_full(i, j);
        }
    }

    // Crear red neuronal con más capacidad
    NeuralNetwork<float> model;
    model.add_layer(std::make_unique<Dense<float>>(9, 12,
        [](auto& w){ w.fill_random_uniform(-1, 1); },
        [](auto& b){ b.fill(0); }));
    model.add_layer(std::make_unique<ReLU<float>>());
    model.add_layer(std::make_unique<Dense<float>>(12, 6,
        [](auto& w){ w.fill_random_uniform(-1, 1); },
        [](auto& b){ b.fill(0); }));
    model.add_layer(std::make_unique<ReLU<float>>());
    model.add_layer(std::make_unique<Dense<float>>(6, 3,
        [](auto& w){ w.fill_random_uniform(-1, 1); },
        [](auto& b){ b.fill(0); }));
    model.add_layer(std::make_unique<Sigmoid<float>>());

    // Entrenar
    model.train<BCELoss>(X_train, Y_train, 500, 32, 0.01);

    // Evaluar sobre datos de prueba
    std::array<std::string, 3> clases = {"X", "O", "Nada"};
    std::cout << "Evaluando sobre conjunto de prueba (" << X_test.shape()[0] << " ejemplos):\n";

    int correctos = 0;
    for (size_t i = 0; i < X_test.shape()[0]; ++i) {
        std::array<size_t, 2> shape_input = {1, 9};
        Tensor<float, 2> input(shape_input);
        for (int j = 0; j < 9; ++j)
            input(0, j) = X_test(i, j);

        auto pred = model.predict(input);
        int clase_predicha = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
        int clase_real = std::distance(&Y_test(i, 0), std::max_element(&Y_test(i, 0), &Y_test(i, 0) + 3));

        if (clase_predicha == clase_real)
            ++correctos;

        std::cout << "Prediccion: " << clases[clase_predicha]
                  << " | Real: " << clases[clase_real] << "\n";
    }

    float accuracy = static_cast<float>(correctos) / X_test.shape()[0];
    std::cout << "\nAccuracy: " << (accuracy * 100.0f) << "%\n";

    return 0;
}