#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#pragma once
#include "tensor.h"
#include <cmath>

namespace utec::neural_network {
// Clase para calcular el error cuadrático medio (Mean Squared Error - MSE)
    template<typename T = double>
    class MSELoss {
        using Tensor2 = utec::algebra::Tensor<T, 2>;
        Tensor2 y_pred, y_true;
    public:
        MSELoss() = default;
        MSELoss(const Tensor2& y_pred, const Tensor2& y_true)
                : y_pred(y_pred), y_true(y_true) {}
        T loss() const {
            T sum = 0;
            for (size_t i = 0; i < y_pred.size(); ++i)
                sum += (y_pred.begin()[i] - y_true.begin()[i]) * (y_pred.begin()[i] - y_true.begin()[i]);
            return sum / y_pred.size();
        }
        Tensor2 loss_gradient() const {
            Tensor2 grad(y_pred.shape());
            for (size_t i = 0; i < y_pred.size(); ++i)
                grad.begin()[i] = 2 * (y_pred.begin()[i] - y_true.begin()[i]) / y_pred.size();
            return grad;
        }
    };
// Clase para calcular la pérdida binaria cruzada (Binary Cross-Entropy - BCE)
    template<typename T = double>
    class BCELoss {
        using Tensor2 = utec::algebra::Tensor<T, 2>;
        Tensor2 y_pred, y_true;
    public:
        BCELoss() = default;
        BCELoss(const Tensor2& y_pred, const Tensor2& y_true)
                : y_pred(y_pred), y_true(y_true) {}
        T loss() const {
            T sum = 0;
            for (size_t i = 0; i < y_pred.size(); ++i) {
                T yp = std::max(std::min(y_pred.begin()[i], T(1) - T(1e-7f)), T(1e-7f));
                sum += -y_true.begin()[i] * std::log(yp) - (1 - y_true.begin()[i]) * std::log(1 - yp);
            }
            return sum / y_pred.size();
        }
        Tensor2 loss_gradient() const {
            Tensor2 grad(y_pred.shape());
            for (size_t i = 0; i < y_pred.size(); ++i) {
                T yp = std::max(std::min(y_pred.begin()[i], T(1) - T(1e-7f)), T(1e-7f));
                grad.begin()[i] = (yp - y_true.begin()[i]) / (yp * (1 - yp) * y_pred.size());
            }
            return grad;
        }
    };
} // namespace utec::neural_network
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H