#ifndef DATASET_CSV_LOADER_H
#define DATASET_CSV_LOADER_H
#include "tensor.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace dataset {
    std::pair<utec::algebra::Tensor<float, 2>, utec::algebra::Tensor<float, 2>>
    load_csv_dataset(const std::string& filename) {
        using Tensor2D = utec::algebra::Tensor<float, 2>;
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> targets;
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            std::vector<float> row;
            while (std::getline(ss, value, ',')) {
                row.push_back(std::stof(value));
            }
            inputs.emplace_back(row.begin(), row.begin() + 9);
            targets.emplace_back(row.begin() + 9, row.end());
        }
        size_t n_samples = inputs.size();
        Tensor2D X(std::array<size_t, 2>{n_samples, 9});
        Tensor2D Y(std::array<size_t, 2>{n_samples, 3});
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < 9; ++j)
                X(i, j) = inputs[i][j];
            for (size_t j = 0; j < 3; ++j)
                Y(i, j) = targets[i][j];
        }
        return {X, Y};
    }
} // namespace dataset
#endif // DATASET_CSV_LOADER_H
