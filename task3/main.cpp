#include <iostream>
#include <mpi/mpi.h>
//#include <mpi.h>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>

using namespace std;


struct Parallelepiped {
    double x;
    double y;
    double z;
};

float getFloatValueFromArg(const char *key, int argc, char *argv[]) {
    float value = 0;
    // Первый параметр - ссылка на сборку
    for (int i = 1; i < argc; ++i) {
        string currentArgument(argv[i]);
        int argName = currentArgument.find(key);
        if (argName != string::npos) {
            string argString = currentArgument.substr(argName + strlen(key));
            try {
                value = atof(argString.c_str());
            } catch (...) {
                // Не поддерживается на кластерах
                // throw runtime_error("Invalid input epsilon");
                throw;
            }
        }
    }

    return value;
}

int getIntValueFromArg(const char *key, int argc, char *argv[]) {
    int value = 0;
    // Первый параметр - ссылка на сборку
    for (int i = 1; i < argc; ++i) {
        string currentArgument(argv[i]);
        int argName = currentArgument.find(key);
        if (argName != string::npos) {
            string argString = currentArgument.substr(argName + strlen(key));
            try {
                value = atoi(argString.c_str());
            } catch (...) {
                // Не поддерживается на кластерах
                // throw runtime_error("Invalid input epsilon");
                throw;
            }
        }
    }

    return value;
}

void initVariables(int argc, char *argv[], double &Lx, double &Ly, double &Lz, double &T, int &N, int &K,
                   double &hx, double &hy, double &hz, double &tau, int &layerSize) {
    // Первый параметр - ссылка на сборку
    Lx = getFloatValueFromArg("-Lx=", argc, argv);
    Ly = getFloatValueFromArg("-Ly=", argc, argv);
    Lz = getFloatValueFromArg("-Lz=", argc, argv);
    T = getFloatValueFromArg("-T=", argc, argv);
    N = getIntValueFromArg("-N=", argc, argv);
    K = getIntValueFromArg("-K=", argc, argv);
    // Остальные параметры будут игнорироваться (ну или позже добавлю какие-нибудь свои кастомные)

    hx = Lx / N;
    hy = Ly / N;
    hz = Lz / N;
    tau = T / K;
    layerSize = pow(N + 1, 3);
}

// Аналитическое решение
double getAnalyticValue(double x, double y, double z, double t, double Lx, double Ly, double Lz) {
    double at = M_PI * sqrt(1 / pow(Lx, 2) + 1 / pow(Ly, 2) + 4 / pow(Lz, 2));

    return sin(M_PI * x / Lx) * sin(M_PI * y / Ly) * sin(2 * z * M_PI / Lz) * cos(at * t + 2 * M_PI);
}

// Одномерная индексация
int getLinearIndex(int i, int j, int k, int N) {
    return (i * (N + 1) + j) * (N + 1) + k;
}

// Заполнение граничными значениями
void fillBoundaryValues(vector<double> &u, double t, bool isInitial, int N, double hx, double hy, double Lx, double Ly,
                        double Lz) {
// #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // X axis
            u.at(getLinearIndex(0, i, j, N)) = 0;
            u.at(getLinearIndex(N, i, j, N)) = 0;

            // Y axis
            u.at(getLinearIndex(i, 0, j, N)) = 0;
            u.at(getLinearIndex(i, N, j, N)) = 0;

            // Z axis
            if (isInitial) {
                u.at(getLinearIndex(i, j, 0, N)) = getAnalyticValue(i * hx, j * hy, 0, t, Lx, Ly, Lz);
                u.at(getLinearIndex(i, j, N, N)) = getAnalyticValue(i * hx, j * hy, Lz, t, Lx, Ly, Lz);
            } else {
                u.at(getLinearIndex(i, j, 0, N)) =
                        (u.at(getLinearIndex(i, j, 1, N)) + u.at(getLinearIndex(i, j, N - 1, N))) / 2;
                u.at(getLinearIndex(i, j, N, N)) =
                        (u.at(getLinearIndex(i, j, 1, N)) + u.at(getLinearIndex(i, j, N - 1, N))) / 2;
            }
        }
    }
}

// Начальные условия
double Phi(double x, double y, double z, double Lx, double Ly, double Lz) {
    return getAnalyticValue(x, y, z, 0, Lx, Ly, Lz);
}

// Оператор Лапласа
double LaplaceOperator(vector<double> u, int i, int j, int k, int N, double hx, double hy, double hz) {
    double temp = 2 * u.at(getLinearIndex(i, j, k, N));
    double dx = (u.at(getLinearIndex(i - 1, j, k, N)) - temp + u.at(getLinearIndex(i + 1, j, k, N))) / (hx * hx);
    double dy = (u.at(getLinearIndex(i, j - 1, k, N)) - temp + u.at(getLinearIndex(i, j + 1, k, N))) / (hy * hy);
    double dz = (u.at(getLinearIndex(i, j, k - 1, N)) - temp + u.at(getLinearIndex(i, j, k + 1, N))) / (hz * hz);
    return dx + dy + dz;
}

// Заполнение начальных условий
void fillVectorByInitialValues(vector<vector<double>> &u, int N, double hx, double hy, double hz, double Lx, double Ly,
                               double Lz, double tau) {
    fillBoundaryValues(u.at(0), 0, true, N, hx, hy, Lx, Ly, Lz);
    fillBoundaryValues(u.at(1), tau, true, N, hx, hy, Lx, Ly, Lz);

// #pragma omp parallel for collapse(3)
    for (int i = 1; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            for (int k = 1; k < N; ++k) {
                u[0][getLinearIndex(i, j, k, N)] = Phi(i * hx, j * hy, k * hz, Lx, Ly, Lz);
            }
        }
    }

// #pragma omp parallel for collapse(3)
    for (int i = 1; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            for (int k = 1; k < N; ++k) {
                u[1][getLinearIndex(i, j, k, N)] =
                        u.at(0).at(getLinearIndex(i, j, k, N)) +
                        tau * tau / 2 * LaplaceOperator(u.at(0), i, j, k, N, hx, hy, hz);
            }
        }
    }
}

// Оценка погрешности на слое
double
EvaluateError(vector<double> u, double t, int N, double hx, double hy, double hz, double Lx, double Ly, double Lz) {
    double error = 0;

// #pragma omp parallel for collapse(3) reduction(max: error)
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            for (int k = 0; k < N; ++k) {
                error = max(error, fabs(u.at(getLinearIndex(i, j, k, N)) -
                                        getAnalyticValue(i * hx, j * hy, k * hz, t, Lx, Ly, Lz)));
            }
        }
    }

    return error;
}

double
makeSolution(double Lx, double Ly, double Lz, int N, double hx, double hy, double hz, double tau,
             int size, int steps) {
    vector<vector<double>> u{vector<double>(size), vector<double>(size), vector<double>(size)};

    fillVectorByInitialValues(u, N, hx, hy, hz, Lx, Ly, Lz, tau);

    cout << "Layer 0 max error: " << EvaluateError(u[0], 0, N, hx, hy, hz, Lx, Ly, Lz) << std::endl;
    cout << "Layer 1 max error: " << EvaluateError(u[1], tau, N, hx, hy, hz, Lx, Ly, Lz) << std::endl;


    for (int step = 2; step <= steps; ++step) {
// #pragma omp parallel for collapse(3)
        for (int i = 1; i < N; ++i) {
            for (int j = 1; j < N; ++j) {
                for (int k = 1; k < N; ++k) {
                    u[step % 3].at(getLinearIndex(i, j, k, N)) = 2 * u[(step + 2) % 3].at(getLinearIndex(i, j, k, N)) -
                                                                 u[(step + 1) % 3].at(getLinearIndex(i, j, k, N)) +
                                                                 tau * tau *
                                                                 LaplaceOperator(u[(step + 2) % 3], i, j, k, N, hx, hy,
                                                                                 hz);

                    fillBoundaryValues(u[step % 3], step * tau, false, N, hx, hy, Lx, Ly, Lz);
                    cout << "Layer " << step << " max error: "
                         << EvaluateError(u[step % 3], step * tau, N, hx, hy, hz, Lx, Ly, Lz) << std::endl;
                }
            }
        }
    }

    double error = EvaluateError(u[steps % 3], steps * tau, N, hx, hy, hz, Lx, Ly, Lz);
    return error;
}

// ЛАЗАРЕВ В.А. / 628 группа / 2 вариант
int main(int argc, char *argv[]) {
    // Размеры параллелепипеда / T - итоговое время
    double Lx, Ly, Lz, T;
    // N - количество точек пространственной сетки / K - количество точек временной сетки
    int N, K;

    int steps = 20;

    // Шаги пространственной сетки по каждой из осей + шаг временной сетки
    double hx, hy, hz, tau;
    // Размер слоя
    int layerSize;

    initVariables(argc, argv, Lx, Ly, Lz, T, N, K, hx, hy, hz, tau, layerSize);

    makeSolution(Lx, Ly, Lz, N, hx, hy, hz, tau, layerSize, steps);

    return 0;
}