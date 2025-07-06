#ifndef DATASET3X3_H
#define DATASET3X3_H
#include "tensor.h"
#include <vector>
namespace dataset {
    using Tensor2D = utec::algebra::Tensor<float, 2>;
    inline std::vector<std::pair<Tensor2D, Tensor2D>> load_dataset() {
        std::vector<std::pair<Tensor2D, Tensor2D>> data;
Tensor2D label_x(std::array<size_t, 2>{1, 3});
Tensor2D label_o(std::array<size_t, 2>{1, 3});
Tensor2D label_n(std::array<size_t, 2>{1, 3});
        std::vector<std::vector<float>> inputs = {
            {1,0,1, 0,1,0, 1,0,1},
            {0.9,0.1,1, 0.2,1,0.1, 0.8,0,0.9},
            {1,0,0.9, 0.1,1,0, 1,0,1},
            {1,1,1, 1,0,1, 1,1,1},
            {0.9,1,0.9, 1,0.1,1, 1,0.9,1},
            {1,1,1, 1,0.05,1, 1,1,1},
            {0,0,1, 0,1,0, 1,0,0},
            {0.2,0.3,0.5, 0.1,0.6,0.1, 0.9,0.2,0.2},
            {1,0,0, 0,0,0, 0,1,0}
        };
        for (int i = 0; i < 9; ++i) {
Tensor<float, 2> input(std::array<size_t, 2>{1, 9});
            for (int j = 0; j < 9; ++j)
                input(0, j) = inputs[i][j];
            if (i < 3)
                data.emplace_back(input, label_x);
            else if (i < 6)
                data.emplace_back(input, label_o);
            else
                data.emplace_back(input, label_n);
        }
        return data;
    }
} // namespace dataset
#endif // DATASET3X3_H
