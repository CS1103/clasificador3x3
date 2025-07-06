[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)

# Proyecto Final 2025-1: Clasificador3x3
## **CS2013 Programación III** · Informe Final

---

### 📌 Descripción

**Clasificador3x3** es un proyecto educativo que implementa desde cero una **red neuronal multicapa en C++20** para clasificar dígitos representados como **matrices 3×3**. Utiliza operaciones vectorizadas con **Eigen** y está completamente modularizado siguiendo buenas prácticas de diseño de software.

---

## Contenidos

1. [Datos generales](#datos-generales)  
2. [Requisitos e instalación](#requisitos-e-instalación)  
3. [Investigación teórica](#1-investigación-teórica)  
4. [Diseño e implementación](#2-diseño-e-implementación)  
5. [Ejecución](#3-ejecución)  
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)  
7. [Trabajo en equipo](#5-trabajo-en-equipo)  
8. [Conclusiones](#6-conclusiones)  
9. [Bibliografía](#7-bibliografía)  
10. [Licencia](#licencia)

---

## Datos generales

- **Tema**: Red neuronal C++ para clasificación de matrices 3x3
- **Grupo**: `React++`
- **Integrantes**:
  - Daniel Sandoval – 209900001 – Implementación de arquitectura, diseño modular, pruebas funcionales y documentación
  - [Completa los demás integrantes según tu grupo real]

---

## Requisitos e instalación

### ✅ Requisitos

- **Compilador**: GCC 11 o superior (C++20)
- **Dependencias**:
  - CMake ≥ 3.18
  - Eigen ≥ 3.4 (https://eigen.tuxfamily.org)

### 🔧 Instalación

```bash
git clone https://github.com/usuario/Clasificador3x3.git
cd Clasificador3x3
mkdir build && cd build
cmake ..
make
````

---

## 1. Investigación teórica

Se investigaron los siguientes temas fundamentales:

* Historia y evolución de las redes neuronales (McCulloch & Pitts, Hebb, Rosenblatt, Rumelhart)
* Arquitecturas relevantes: MLP, CNN y RNN
* Algoritmos de entrenamiento: backpropagation y optimizadores modernos (SGD, Adam)
* Funciones de activación: Sigmoid, ReLU, Softmax
* Funciones de pérdida: MSE, Cross-Entropy

Ver [bibliografía](#7-bibliografía).

---

## 2. Diseño e implementación

### 🧠 Arquitectura

* **Tensor\<T, R>**: clase genérica para representar tensores de dimensión arbitraria.
* **Capas modulares**: Dense, Activación (ReLU, Sigmoid), Pérdida, Optimización.
* **Red neuronal** (`NeuralNetwork`): estructura tipo MLP configurable desde `main.cpp`.

### 🧩 Patrones de diseño

* **Polimorfismo** en capas (`Layer` abstracta).
* **Strategy pattern** para pérdidas (`ILoss`) y optimizadores (`IOptimizer`).
* **Modularización** en headers por responsabilidad (`nn_dense.h`, `nn_loss.h`, etc.).

### 📁 Estructura del proyecto

```
Clasificador3x3/
├── include/
│   ├── tensor.h
│   ├── nn_dense.h
│   ├── nn_activation.h
│   ├── nn_loss.h
│   ├── nn_optimizer.h
│   └── neural_network.h
├── src/
│   ├── main.cpp
│   ├── nn_dense.cpp
│   ├── nn_activation.cpp
│   ├── nn_loss.cpp
│   ├── nn_optimizer.cpp
│   └── neural_network.cpp
├── test/
│   └── test_network.cpp
├── data/
│   └── input.csv
├── CMakeLists.txt
├── README.md
└── LICENSE
```

---

## 3. Ejecución

```bash
./build/clasificador input.csv output.csv
```

* El programa leerá los datos de entrada desde `input.csv`, entrenará la red neuronal y almacenará los resultados de clasificación en `output.csv`.
* Puedes modificar los datos sin necesidad de reescribir código fuente, permitiendo autonomía total.

---

## 4. Análisis del rendimiento

* **Precisión**: +95% en entrenamiento con dataset pequeño de ejemplos binarios 3x3.
* **Velocidad**: entrenamiento completo en milisegundos.
* **Escalabilidad**: soporta expansión con más datos gracias a Eigen y diseño eficiente.

### ⚠️ Limitaciones

* Arquitectura básica (solo una capa oculta).
* Dataset mínimo (poca capacidad de generalización).
* Uso académico, no está preparado para datasets reales (p. ej., MNIST 28x28).

---

## 5. Trabajo en equipo

| Tarea                   | Integrante      | Rol                        |
| ----------------------- | --------------- | -------------------------- |
| Investigación teórica   | Alumno A        | Redacción teórica          |
| Diseño arquitectónico   | Alumno B        | UML, estructura modular    |
| Implementación modelo   | Daniel Sandoval | Implementación del código  |
| Pruebas y benchmarking  | Alumno D        | Métricas y testeo          |
| Documentación y entrega | Alumno E        | README, presentación final |

---

## 6. Conclusiones

* ✅ Se desarrolló una red neuronal funcional en C++ puro, desde cero.
* 🧱 Se aplicaron principios de diseño modular y reutilizable.
* 💡 Se validaron conceptos clave de aprendizaje automático, programación genérica y estructuras avanzadas de C++.
* 📊 Se obtuvo alta precisión en pruebas simples, validando el enfoque.

---

## 7. Bibliografía

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep Learning. *Nature*.
2. Rumelhart, D., Hinton, G., & Williams, R. (1986). Learning representations by back-propagating errors. *Nature*.
3. Schmidhuber, J. (2015). Deep Learning in Neural Networks: An Overview. *Neural Networks*.
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más información.
