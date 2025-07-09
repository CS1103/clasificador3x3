# Proyecto Final 2025-1: Clasificador3x3

**CS2013 Programación III · Informe Final**

## 📌 Descripción

Implementación de una red neuronal multicapa en C++20 desde cero para la clasificación de patrones simples representados como matrices binarias de 3×3, correspondientes a las clases “X”, “O” y “Nada”.

---

## Contenidos

* Datos generales
* Requisitos e instalación
* Investigación teórica
* Diseño e implementación
* Ejecución
* Análisis del rendimiento
* Trabajo en equipo
* Conclusiones
* Bibliografía
* Licencia

---

## Datos generales

* **Tema:** Redes Neuronales en AI
* **Grupo:** React++

### Integrantes:

* **Daniel Guillermo Sandoval Toro– 202310533** (Implementación de la arquitectura, modelo final)
* **Sergio Leonardo Llanos Parraga– 202210188** (Documentación y demo)


---

## Requisitos e instalación

**Compilador:**

* GCC 11 o superior (soporte completo de C++20)

**Dependencias:**

* CMake 3.18 o superior
* Eigen 3.4 ([https://eigen.tuxfamily.org](https://eigen.tuxfamily.org))

### Instalación

```bash
git clone https://github.com/CS1103/clasificador3x3.git
cd clasificador3x3
mkdir build && cd build
cmake ..
make
```

---

## 1. Investigación teórica

**Objetivo:** Explorar fundamentos y arquitecturas de redes neuronales artificiales (ANNs) y aplicarlas en C++.

**Contenido:**

* Historia y evolución de las redes neuronales: McCulloch & Pitts, Perceptrón de Rosenblatt, Rumelhart y el algoritmo de retropropagación.
* Arquitecturas relevantes: MLP (Multilayer Perceptron), comparación con CNNs y RNNs.
* Algoritmos de entrenamiento: Backpropagation, Gradient Descent.
* Funciones de activación: ReLU, Sigmoid.
* Funciones de pérdida: Binary Cross Entropy (BCE), Mean Squared Error (MSE).

---

## 2. Diseño e implementación

### 2.1 Arquitectura de la solución

**Componentes principales:**

* `Tensor<T, R>`: clase genérica para estructuras de datos N-dimensionales con operaciones matemáticas vectorizadas (broadcasting, slicing, reducción).
* `Dense`: capa totalmente conectada (fully connected) con pesos y bias entrenables.
* `ReLU`, `Sigmoid`: funciones de activación.
* `NeuralNetwork`: clase modular que orquesta capas y funciones de pérdida.
* `BCELoss`, `MSELoss`: funciones de pérdida derivadas de `ILoss`.
* `SGD`, `Adam`: optimizadores modulares derivados de `IOptimizer`.

**Patrones de diseño:**

* **Polimorfismo**: para capas, funciones de pérdida y optimizadores.
* **Strategy pattern**: para la selección de funciones de pérdida y optimización.
* **Modularidad por cabeceras**: `nn_activation.h`, `nn_loss.h`, `nn_optimizer.h`, etc.

**Estructura de carpetas:**

```
clasificador3x3/
├── include/           # Archivos cabecera (.h)
│   ├── tensor.h
│   ├── nn_dense.h
│   ├── nn_activation.h
│   ├── nn_loss.h
│   ├── nn_optimizer.h
│   ├── nn_interfaces.h
│   └── neural_network.h
├── src/               # Código fuente (.cpp)
│   ├── main.cpp
├── data/
│   └── dataset3x3.csv
├── CMakeLists.txt
└── README.md
```

---

### 2.2 Manual de uso y casos de prueba

**Ejecución desde terminal:**

```bash
./clasificador3x3
```

**Entrada esperada:**
Archivo CSV `dataset3x3.csv` con ejemplos de entrenamiento con 9 valores binarios por fila (3x3) y etiqueta one-hot (3 columnas).

**Casos de prueba:**

* Carga y parsing de dataset CSV
* Test de capa `Dense` con propagación hacia adelante
* Test de función de activación `ReLU`
* Validación de predicción final comparando `argmax` del output con la clase real
* Precisión sobre conjunto de prueba (`accuracy`)

---

## 3. Ejecución

**Flujo:**

1. Se carga el dataset 3x3 desde `dataset3x3.csv`
2. Se divide automáticamente en 80% entrenamiento y 20% prueba
3. La red se construye y entrena por 500 épocas con `BCELoss`
4. Se imprime por consola la clase predicha vs. real y precisión final

**Ejemplo de salida:**

```
Prediccion: O | Real: O  
Prediccion: Nada | Real: Nada  
...
Accuracy: 90.0%
```

---

## 4. Análisis del rendimiento

**Métricas:**

* Iteraciones: 500 épocas
* Tiempo total de entrenamiento: < 1 segundo (dataset pequeño)
* Precisión final: entre **85% y 92%**

**Ventajas:**

* Modularidad total del código
* Código C++ puro y ligero, sin dependencias pesadas
* Entrenamiento rápido en CPU

**Desventajas:**

* Dataset muy limitado (figuras simples)
* No optimizado para procesamiento en paralelo
* No generaliza a imágenes reales o tareas complejas

**Mejoras futuras:**

* Reemplazar multiplicaciones por BLAS/Eigen para mayor velocidad
* Permitir entrenamiento multi-thread con OpenMP
* Soporte para Softmax y clasificación multiclase más robusta

---

## 5. Trabajo en equipo

| Tarea                     | Miembro             | Rol                                       |
| ------------------------- | ------------------- | ----------------------------------------- |
| Investigación teórica     | **Daniel Sandoval** | Documentar bases teóricas                 |
| Diseño de la arquitectura | **Daniel Sandoval** | UML y estructura de clases                |
| Implementación del modelo | **Daniel Sandoval** | Programación del sistema completo         |
| Pruebas y benchmarking    | sergio              | Generación de métricas y validación       |
| Documentación y demo      | sergio              | Redacción del README y video de ejecución |

*Actualizar con los nombres reales si aplica*

---

## 6. Conclusiones

* **Logros:**

  * Implementación completa de red neuronal desde cero en C++
  * Modularidad, reutilización y claridad en el diseño
  * Precisión adecuada en pruebas sintéticas

* **Evaluación:**

  * El rendimiento cumple con los objetivos académicos del curso
  * Buen manejo de memoria y eficiencia en ejecución

* **Aprendizajes:**

  * Retropropagación y descenso de gradiente
  * Diseño de software moderno en C++
  * Uso de estructuras de datos genéricas y abstracción

* **Recomendaciones:**

  * Explorar datasets reales (MNIST)
  * Añadir visualizaciones de entrenamiento y métricas

---

## 7. Bibliografía

\[1] Y. LeCun, Y. Bengio y G. Hinton, “Deep learning,” *Nature*, vol. 521, pp. 436–444, 2015.
\[2] D. E. Rumelhart, G. E. Hinton y R. J. Williams, “Learning representations by back-propagating errors,” *Nature*, vol. 323, pp. 533–536, 1986.
\[3] J. Schmidhuber, “Deep Learning in Neural Networks: An Overview,” *Neural Networks*, vol. 61, pp. 85–117, 2015.
\[4] I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, MIT Press, 2016.

---

## 8. Video de la demo

Link de la demo del video: 

---

## 9. Link del informe

Hipervinculo del informe: 

---
## Licencia

Este proyecto usa la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
