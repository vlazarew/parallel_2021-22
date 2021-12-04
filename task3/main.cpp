#include <iostream>
#include <mpi/mpi.h>
//#include <mpi.h>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>
#include <omp.h>

using namespace std;

// Исходный параллелпипед
struct Parallelepiped {
    // Размеры параллелепипеда по каждой из осей
    double x;
    double y;
    double z;
};

// Шаги пространственной сетки по каждой из осей
struct GridSteps {
    double x;
    double y;
    double z;
};

struct SolverVariables {
    // Исходный параллелпипед
    Parallelepiped L;
    // T - итоговое время
    double T;
    // N - количество точек пространственной сетки / K - количество точек временной сетки / steps - кол-во шагов для решения
    int N, K, steps;

    // Шаги пространственной сетки по каждой из осей
    GridSteps H;
    // Шаг временной сетки
    double tau;
    // Размер слоя
    int layerSize;

    // Количество нитей OMP
    int ompThreadsCount;

    // Id процесса / количество MPI-процессов
    int processId, countOfProcesses;
};

float getFloatValueFromArg(const char *key, int argc, char *argv[], float defaultValue) {
    float value = 0;
    bool valueFound = false;
    // Первый параметр - ссылка на сборку
    for (int i = 1; i < argc; ++i) {
        string currentArgument(argv[i]);
        int argName = currentArgument.find(key);
        if (argName != string::npos) {
            string argString = currentArgument.substr(argName + strlen(key));
            try {
                value = atof(argString.c_str());
                valueFound = true;
                break;
            } catch (...) {
                // Не поддерживается на кластерах
                // throw runtime_error("Invalid input epsilon");
                throw;
            }
        }
    }

    return valueFound ? value : defaultValue;
}

int getIntValueFromArg(const char *key, int argc, char *argv[], int defaultValue) {
    int value = 0;
    bool valueFound = false;
    // Первый параметр - ссылка на сборку
    for (int i = 1; i < argc; ++i) {
        string currentArgument(argv[i]);
        int argName = currentArgument.find(key);
        if (argName != string::npos) {
            string argString = currentArgument.substr(argName + strlen(key));
            try {
                value = atoi(argString.c_str());
                valueFound = true;
                break;
            } catch (...) {
                // Не поддерживается на кластерах
                // throw runtime_error("Invalid input epsilon");
                throw;
            }
        }
    }

    return valueFound ? value : defaultValue;
}

void initVariables(int argc, char *argv[], SolverVariables &variables) {
    // Первый параметр - ссылка на сборку
    variables.L.x = getFloatValueFromArg("-Lx=", argc, argv, 1);
    variables.L.y = getFloatValueFromArg("-Ly=", argc, argv, 1);
    variables.L.z = getFloatValueFromArg("-Lz=", argc, argv, 1);
    variables.T = getFloatValueFromArg("-T=", argc, argv, 1);
    variables.N = getIntValueFromArg("-N=", argc, argv, 40);
    variables.K = getIntValueFromArg("-K=", argc, argv, 100);
    variables.steps = getIntValueFromArg("-steps=", argc, argv, 20);
    variables.ompThreadsCount = getIntValueFromArg("-omp=", argc, argv, 1);
    // Остальные параметры будут игнорироваться (ну или позже добавлю какие-нибудь свои кастомные)

    variables.H.x = variables.L.x / variables.N;
    variables.H.y = variables.L.y / variables.N;
    variables.H.z = variables.L.z / variables.N;
    variables.tau = variables.T / variables.K;
    variables.layerSize = pow(variables.N + 1, 3);
}

// Аналитическое решение
double getAnalyticValue(double x, double y, double z, double t, Parallelepiped L) {
    double at = M_PI * sqrt(1 / pow(L.x, 2) + 1 / pow(L.y, 2) + 4 / pow(L.z, 2));

    return sin(M_PI * x / L.x) * sin(M_PI * y / L.y) * sin(2 * z * M_PI / L.z) * cos(at * t + 2 * M_PI);
}

// Одномерная индексация
int getLinearIndex(int i, int j, int k, int N) {
    return (i * (N + 1) + j) * (N + 1) + k;
}

// Заполнение граничными значениями
void fillBoundaryValues(vector<double> &u, double tau, bool isInitial, SolverVariables variables) {
    int N = variables.N;
    double hx = variables.H.x;
    double hy = variables.H.y;

#pragma omp parallel for collapse(2)
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
                u.at(getLinearIndex(i, j, 0, N)) = getAnalyticValue(i * hx, j * hy, 0, tau, variables.L);
                u.at(getLinearIndex(i, j, N, N)) = getAnalyticValue(i * hx, j * hy, variables.L.z, tau, variables.L);
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
double Phi(double x, double y, double z, Parallelepiped L) {
    return getAnalyticValue(x, y, z, 0, L);
}

// Оператор Лапласа
double LaplaceOperator(vector<double> u, int i, int j, int k, int N, GridSteps H) {
    double temp = 2 * u.at(getLinearIndex(i, j, k, N));
    double dx = (u.at(getLinearIndex(i - 1, j, k, N)) - temp + u.at(getLinearIndex(i + 1, j, k, N))) / (H.x * H.x);
    double dy = (u.at(getLinearIndex(i, j - 1, k, N)) - temp + u.at(getLinearIndex(i, j + 1, k, N))) / (H.y * H.y);
    double dz = (u.at(getLinearIndex(i, j, k - 1, N)) - temp + u.at(getLinearIndex(i, j, k + 1, N))) / (H.z * H.z);
    return dx + dy + dz;
}

// Заполнение начальных условий
void fillVectorByInitialValues(vector<vector<double>> &u, SolverVariables variables) {
    fillBoundaryValues(u.at(0), 0, true, variables);
    fillBoundaryValues(u.at(1), variables.tau, true, variables);

    int N = variables.N;
    GridSteps H = variables.H;

#pragma omp parallel for collapse(3)
    for (int i = 1; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            for (int k = 1; k < N; ++k) {
                u[0][getLinearIndex(i, j, k, N)] = Phi(i * H.x, j * H.y, k * H.z, variables.L);
            }
        }
    }

#pragma omp parallel for collapse(3)
    for (int i = 1; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            for (int k = 1; k < N; ++k) {
                u[1][getLinearIndex(i, j, k, N)] =
                        u.at(0).at(getLinearIndex(i, j, k, N)) +
                        variables.tau * variables.tau / 2 * LaplaceOperator(u.at(0), i, j, k, N, H);
            }
        }
    }
}

// Оценка погрешности на слое
double
EvaluateError(vector<double> u, double t, SolverVariables variables) {
    double error = 0;
    int N = variables.N;

#pragma omp parallel for collapse(3) reduction(max: error)
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            for (int k = 0; k < N; ++k) {
                error = max(error, fabs(u.at(getLinearIndex(i, j, k, N)) -
                                        getAnalyticValue(i * variables.H.x, j * variables.H.y, k * variables.H.z, t,
                                                         variables.L)));
            }
        }
    }

    return error;
}

double
makeSolution(SolverVariables variables) {
    int layerSize = variables.layerSize;
    int steps = variables.steps;
    int N = variables.N;
    double tau = variables.tau;

    vector<vector<double>> u{vector<double>(layerSize), vector<double>(layerSize), vector<double>(layerSize)};

    fillVectorByInitialValues(u, variables);

    cout << "Layer 0 max error: " << EvaluateError(u[0], 0, variables) << std::endl;
    cout << "Layer 1 max error: " << EvaluateError(u[1], variables.tau, variables) << std::endl;


    for (int step = 2; step <= steps; ++step) {
#pragma omp parallel for collapse(3)
        for (int i = 1; i < N; ++i) {
            for (int j = 1; j < N; ++j) {
                for (int k = 1; k < N; ++k) {
                    u[step % 3].at(getLinearIndex(i, j, k, N)) = 2 * u[(step + 2) % 3].at(getLinearIndex(i, j, k, N)) -
                                                                 u[(step + 1) % 3].at(getLinearIndex(i, j, k, N)) +
                                                                 tau * tau *
                                                                 LaplaceOperator(u[(step + 2) % 3], i, j, k, N,
                                                                                 variables.H);
                }
            }
        }
        fillBoundaryValues(u[step % 3], step * tau, false, variables);
        cout << "Layer " << step << " max error: "
             << EvaluateError(u[step % 3], step * tau, variables) << std::endl;
    }

    double error = EvaluateError(u[steps % 3], steps * tau, variables);
    return error;
}

// ЛАЗАРЕВ В.А. / 628 группа / 2 вариант
int main(int argc, char *argv[]) {
    SolverVariables variables{};

    // Инициализация MPI, создание группы процессов, создание области связи MPI_COMM_WORLD
    MPI_Init(nullptr, nullptr);

    initVariables(argc, argv, variables);

    omp_set_num_threads(variables.ompThreadsCount);

    double start = MPI_Wtime();

    double error = makeSolution(variables);

    double end = MPI_Wtime();
    double diffTime = end - start;

    cout << "OMP threads: " << variables.ompThreadsCount << endl;
    cout << "Final error: " << error << endl;
    cout << "Total time (s): " << diffTime << endl;

    return 0;
}