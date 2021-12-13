#include <iostream>
#include <fstream>
#include <mpi/mpi.h>
//#include <mpi.h>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>
#include <omp.h>

using namespace std;

#pragma region Entities and constants
const int MAIN_PROCESS_ID = 0;

enum Axis {
    X, Y, Z,
};

// Исходный параллелпипед
struct Parallelepiped {
    // Размеры параллелепипеда по каждой из осей
    double x;
    double y;
    double z;
};

// Параллелпипед, обрабтываемый отдельным процессом
struct ProcessParallelepiped {
    // Границы по оси X
    int xMin, xMax;

    // Границы по оси Y
    int yMin, yMax;

    // Границы по оси Z
    int zMin, zMax;

    // Размеры по каждой из осей + объем области
    int dx, dy, dz, size;
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

    // Параллелпипед, обрабтываемый отдельным процессом
    ProcessParallelepiped processParallelepiped;
    // Параллелепипеды-соседи на передачу
    vector<ProcessParallelepiped> send;
    // Параллелепипеды-соседи на прием
    vector<ProcessParallelepiped> recv;
    // Параллелепипеды-соседи в формате id процесса
    vector<int> processIds;

};
#pragma endregion

#pragma region Handle and init arguments

float getFloatValueFromArg(const char *key, int argc, char *argv[], float defaultValue) {
    float value = 0;
    bool valueFound = false;
    // Первый параметр - ссылка на сборку
    for (int i = 1; i < argc; i++) {
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
    for (int i = 1; i < argc; i++) {
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

void initVariables(int argc, char *argv[], SolverVariables &variables, int processId, int countOfProcesses) {
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

    variables.processId = processId;
    variables.countOfProcesses = countOfProcesses;
}

#pragma endregion

#pragma region Split parallelepiped

ProcessParallelepiped createParallelepiped(int xMin, int xMax, int yMin, int yMax, int zMin, int zMax) {
    int dx = xMax - xMin + 1;
    int dy = yMax - yMin + 1;
    int dz = zMax - zMin + 1;
    int size = dx * dy * dz;

    return ProcessParallelepiped{xMin, xMax, yMin, yMax, zMin, zMax, dx, dy, dz, size};
}

// Блочное разбиение параллелепипеда
void makeSplit(int xMin, int xMax, int yMin, int yMax, int zMin, int zMax, int countOfProcesses, Axis axis,
               vector<ProcessParallelepiped> &subParallelepipeds) {

    // Если один процесс, то ему достается полный параллелепипед
    if (countOfProcesses == 1) {
        subParallelepipeds.push_back(createParallelepiped(xMin, xMax, yMin, yMax, zMin, zMax));
        return;
    }

    if (countOfProcesses % 2 == 1) {
        if (axis == X) {
            int newXMax = xMin + (xMax - xMin) / countOfProcesses;
            subParallelepipeds.push_back(createParallelepiped(xMin, newXMax, yMin, yMax, zMin, zMax));

            xMin = newXMax + 1;
            axis = Y;
        } else if (axis == Y) {
            int newYMax = yMin + (yMax - yMin) / countOfProcesses;
            subParallelepipeds.push_back(createParallelepiped(xMin, xMax, yMin, newYMax, zMin, zMax));

            yMin = newYMax + 1;
            axis = Z;
        } else if (axis == Z) {
            int newZMax = zMin + (zMax - zMin) / countOfProcesses;
            subParallelepipeds.push_back(createParallelepiped(xMin, xMax, yMin, yMax, zMin, newZMax));

            zMin = newZMax + 1;
            axis = X;
        }

        countOfProcesses--;
    }

    int newCountOfProcesses = countOfProcesses / 2;
    if (axis == X) {
        int newXMax = (xMin + xMax) / 2;
        makeSplit(xMin, newXMax, yMin, yMax, zMin, zMax, newCountOfProcesses, Y, subParallelepipeds);
        makeSplit(newXMax + 1, xMax, yMin, yMax, zMin, zMax, newCountOfProcesses, Y, subParallelepipeds);
    } else if (axis == Y) {
        int newYMax = (yMin + yMax) / 2;
        makeSplit(xMin, xMax, yMin, newYMax, zMin, zMax, newCountOfProcesses, Z, subParallelepipeds);
        makeSplit(xMin, xMax, newYMax + 1, yMax, zMin, zMax, newCountOfProcesses, Z, subParallelepipeds);
    } else if (axis == Z) {
        int newZMax = (zMin + zMax) / 2;
        makeSplit(xMin, xMax, yMin, yMax, zMin, newZMax, newCountOfProcesses, X, subParallelepipeds);
        makeSplit(xMin, xMax, yMin, yMax, newZMax + 1, zMax, newCountOfProcesses, X, subParallelepipeds);
    }

}

vector<ProcessParallelepiped> splitParallelepiped(int N, int countOfProcesses) {
    vector<ProcessParallelepiped> subParallelepipeds;

    makeSplit(0, N, 0, N, 0, N, countOfProcesses, X, subParallelepipeds);

    return subParallelepipeds;
}

#pragma endregion

#pragma region Actions with neighbours

bool isInside(int xMin1, int xMax1, int yMin1, int yMax1, int xMin2, int xMax2, int yMin2, int yMax2) {
    return xMin2 <= xMin1 && xMax1 <= xMax2 && yMin2 <= yMin1 && yMax1 <= yMax2;
}

bool getNeighbours(ProcessParallelepiped first, ProcessParallelepiped second, ProcessParallelepiped &result) {
    if (first.xMin == second.xMax + 1 || second.xMin == first.xMax + 1) {
        int x = (first.xMin == second.xMax + 1) ? first.xMin : first.xMax;

        if (isInside(first.yMin, first.yMax, first.zMin, first.zMax, second.yMin, second.yMax, second.zMin,
                     second.zMax)) {
            result = createParallelepiped(x, x, first.yMin, first.yMax, first.zMin, first.zMax);
            return true;
        }

        if (isInside(second.yMin, second.yMax, second.zMin, second.zMax, first.yMin, first.yMax, first.zMin,
                     first.zMax)) {
            result = createParallelepiped(x, x, second.yMin, second.yMax, second.zMin, second.zMax);
            return true;
        }

        return false;
    }

    if (first.yMin == second.yMax + 1 || second.yMin == first.yMax + 1) {
        int y = (first.yMin == second.yMax + 1) ? first.yMin : first.yMax;

        if (isInside(first.xMin, first.xMax, first.zMin, first.zMax, second.xMin, second.xMax, second.zMin,
                     second.zMax)) {
            result = createParallelepiped(first.xMin, first.xMax, y, y, first.zMin, first.zMax);
            return true;
        }

        if (isInside(second.xMin, second.xMax, second.zMin, second.zMax, first.xMin, first.xMax, first.zMin,
                     first.zMax)) {
            result = createParallelepiped(second.xMin, second.xMax, y, y, second.zMin, second.zMax);
            return true;
        }

        return false;
    }

    if (first.zMin == second.zMax + 1 || second.zMin == first.zMax + 1) {
        int z = (first.zMin == second.zMax + 1) ? first.zMin : first.zMax;

        if (isInside(first.xMin, first.xMax, first.yMin, first.yMax, second.xMin, second.xMax, second.yMin,
                     second.yMax)) {
            result = createParallelepiped(first.xMin, first.xMax, first.yMin, first.yMax, z, z);
            return true;
        }

        if (isInside(second.xMin, second.xMax, second.yMin, second.yMax, first.xMin, first.xMax, first.yMin,
                     first.yMax)) {
            result = createParallelepiped(second.xMin, second.xMax, second.yMin, second.yMax, z, z);
            return true;
        }

        return false;
    }

    return false;
}

void fillNeighbours(vector<ProcessParallelepiped> &parallelepipeds, SolverVariables &variables) {
    variables.send.clear();
    variables.recv.clear();
    variables.processIds.clear();
    ProcessParallelepiped targetParallelepiped = variables.processParallelepiped;

    for (int i = 0; i < variables.countOfProcesses; i++) {
        if (i == variables.processId) {
            continue;
        }

        ProcessParallelepiped send{};
        ProcessParallelepiped recv{};

        ProcessParallelepiped &processParallelepiped = parallelepipeds[i];
        if (!getNeighbours(targetParallelepiped, processParallelepiped, send)) {
            continue;
        }

        getNeighbours(processParallelepiped, targetParallelepiped, recv);
        variables.processIds.push_back(i);
        variables.send.push_back(send);
        variables.recv.push_back(recv);
    }
}

#pragma endregion

#pragma region Math calculations
// Аналитическое решение
double getAnalyticValue(double x, double y, double z, double t, Parallelepiped L) {
    double at = M_PI * sqrt(1 / pow(L.x, 2) + 1 / pow(L.y, 2) + 4 / pow(L.z, 2));

    return sin(M_PI * x / L.x) * sin(M_PI * y / L.y) * sin(2 * z * M_PI / L.z) * cos(at * t + 2 * M_PI);
}

// Начальные условия
double getPhi(double x, double y, double z, Parallelepiped L) {
    return getAnalyticValue(x, y, z, 0, L);
}

int getIndex(int x, int y, int z, ProcessParallelepiped target) {
    return (x - target.xMin) * target.dy * target.dz + (y - target.yMin) * target.dz + (z - target.zMin);
}

int getLocalIndex(int x, int y, int z, const SolverVariables &variables) {
    return getIndex(x, y, z, variables.processParallelepiped);
}

double findValue(vector<double> u, int i, int j, int k, vector<vector<double>> recv, SolverVariables variables) {
    for (int index = 0; index < variables.processIds.size(); index++) {
        ProcessParallelepiped parallelepiped = variables.recv[index];

        if (i < parallelepiped.xMin || i > parallelepiped.xMax ||
            i < parallelepiped.yMin || i > parallelepiped.yMax ||
            i < parallelepiped.zMin || i > parallelepiped.zMax) {
            continue;
        }

        return recv[index][getIndex(i, j, k, parallelepiped)];
    }

    return u[getLocalIndex(i, j, k, variables)];
}

// Оператор Лапласа
double calculateLaplaceOperator(vector<double> u, int i, int j, int k, const vector<vector<double>> &recv, GridSteps H,
                                const SolverVariables &variables) {
    double dx = (findValue(u, i - 1, j, k, recv, variables) + findValue(u, i + 1, j, k, recv, variables) -
                 2 * u[getLocalIndex(i, j, k, variables)]) / (H.x * H.x);
    double dy = (findValue(u, i, j - 1, k, recv, variables) + findValue(u, i, j + 1, k, recv, variables) -
                 2 * u[getLocalIndex(i, j, k, variables)]) / (H.y * H.y);
    double dz = (findValue(u, i, j, k - 1, recv, variables) + findValue(u, i, j, k + 1, recv, variables) -
                 2 * u[getLocalIndex(i, j, k, variables)]) / (H.z * H.z);

    return dx + dy + dz;
}

double getBoundaryValue(int x, int y, int z, double t, const SolverVariables &variables) {
    int N = variables.N;
    double hx = variables.H.x;
    double hy = variables.H.y;
    double hz = variables.H.z;

    if (x == 0 || x == N) {
        return 0;
    }

    if (y == 0 || y == N) {
        return 0;
    }

    if (z == 0 || z == N) {
        return getAnalyticValue(x * hx, y * hy, z * hz, t, variables.L);
    }

    return 0;
}

// Заполнение граничными значениями
void fillBoundaryValues(vector<double> &u, double tau, const SolverVariables &variables) {
    int N = variables.N;

    ProcessParallelepiped processParallelepiped = variables.processParallelepiped;
    if (processParallelepiped.xMin == 0) {
#pragma omp parallel for collapse(2)
        for (int i = processParallelepiped.yMin; i < processParallelepiped.yMax; i++) {
            for (int j = processParallelepiped.zMin; j < processParallelepiped.zMax; j++) {
                u[getLocalIndex(processParallelepiped.xMin, i, j, variables)] =
                        getBoundaryValue(processParallelepiped.xMin, i, j, tau, variables);
            }
        }
    }

    if (processParallelepiped.xMax == N) {
#pragma omp parallel for collapse(2)
        for (int i = processParallelepiped.yMin; i < processParallelepiped.yMax; i++) {
            for (int j = processParallelepiped.zMin; j < processParallelepiped.zMax; j++) {
                u[getLocalIndex(processParallelepiped.xMax, i, j, variables)] =
                        getBoundaryValue(processParallelepiped.xMax, i, j, tau, variables);
            }
        }
    }

    if (processParallelepiped.yMin == 0) {
#pragma omp parallel for collapse(2)
        for (int i = processParallelepiped.xMin; i <= processParallelepiped.xMax; i++) {
            for (int j = processParallelepiped.zMin; j <= processParallelepiped.zMax; j++) {
                u[getLocalIndex(i, processParallelepiped.yMin, j, variables)] =
                        getBoundaryValue(i, processParallelepiped.yMin, j, tau, variables);
            }
        }
    }

    if (processParallelepiped.yMax == N) {
#pragma omp parallel for collapse(2)
        for (int i = processParallelepiped.xMin; i <= processParallelepiped.xMax; i++) {
            for (int j = processParallelepiped.zMin; j <= processParallelepiped.zMax; j++) {
                u[getLocalIndex(i, processParallelepiped.yMax, j, variables)] =
                        getBoundaryValue(i, processParallelepiped.yMax, j, tau, variables);
            }
        }
    }

    if (processParallelepiped.zMin == 0) {
#pragma omp parallel for collapse(2)
        for (int i = processParallelepiped.xMin; i <= processParallelepiped.xMax; i++) {
            for (int j = processParallelepiped.yMin; j <= processParallelepiped.yMax; j++) {
                u[getLocalIndex(i, j, processParallelepiped.zMin, variables)] =
                        getBoundaryValue(i, j, processParallelepiped.zMin, tau, variables);
            }
        }
    }

    if (processParallelepiped.zMax == N) {
#pragma omp parallel for collapse(2)
        for (int i = processParallelepiped.xMin; i <= processParallelepiped.xMax; i++)
            for (int j = processParallelepiped.yMin; j <= processParallelepiped.yMax; j++)
                u[getLocalIndex(i, j, processParallelepiped.zMax, variables)] =
                        getBoundaryValue(i, j, processParallelepiped.zMax, tau, variables);
    }
}
#pragma endregion

#pragma region data send recv

vector<double>
packParallelepiped(vector<double> u, ProcessParallelepiped parallelepiped, const SolverVariables &variables) {
    vector<double> packed(parallelepiped.size);

#pragma omp parallel for collapse(3)
    for (int i = parallelepiped.xMin; i <= parallelepiped.xMax; i++) {
        for (int j = parallelepiped.yMin; j <= parallelepiped.yMax; j++) {
            for (int k = parallelepiped.zMin; k <= parallelepiped.zMax; k++) {
                packed[getIndex(i, j, k, parallelepiped)] = u[getLocalIndex(i, j, k, variables)];
            }
        }
    }

    return packed;
}

vector<vector<double>> sendRecvValues(const vector<double> &u, SolverVariables variables) {
    unsigned long countOfNeighbours = variables.processIds.size();
    vector<vector<double>> recv(countOfNeighbours);

    for (int i = 0; i < countOfNeighbours; i++) {
        vector<double> packed = packParallelepiped(u, variables.send[i], variables);
        recv[i] = vector<double>(variables.recv[i].size);

        vector<MPI_Request> requests(2);
        vector<MPI_Status> statuses(2);

        MPI_Isend(packed.data(), variables.send[i].size, MPI_DOUBLE, variables.processIds[i], 0, MPI_COMM_WORLD,
                  &requests[0]);
        MPI_Irecv(recv[i].data(), variables.recv[i].size, MPI_DOUBLE, variables.processIds[i], 0,
                  MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests.data(), statuses.data());
    }

    return recv;
}

vector<double>
sendRecvTotal(vector<double> u, vector<ProcessParallelepiped> parallelepipeds, const SolverVariables &variables) {
    if (variables.processId != 0) {
        MPI_Request request;
        MPI_Status status;

        MPI_Isend(u.data(), variables.processParallelepiped.size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Waitall(1, &request, &status);
        return u;
    }

    vector<double> uAll(variables.layerSize);
    ProcessParallelepiped parallelepipedAll = createParallelepiped(0, variables.N, 0, variables.N, 0, variables.N);

    for (int index = 0; index < variables.countOfProcesses; index++) {
        ProcessParallelepiped &parallelepiped = parallelepipeds[index];
        vector<double> uI(parallelepiped.size);

        if (index == variables.processId) {
            uI = u;
        } else {
            vector<MPI_Request> requests(1);
            vector<MPI_Status> statuses(1);

            MPI_Irecv(uI.data(), parallelepiped.size, MPI_DOUBLE, index, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Waitall(1, requests.data(), statuses.data());
        }

        for (int i = parallelepiped.xMin; i <= parallelepiped.xMax; i++) {
            for (int j = parallelepiped.yMin; j <= parallelepiped.yMax; j++) {
                for (int k = parallelepiped.zMin; k <= parallelepiped.zMax; k++) {
                    uAll[getIndex(i, j, k, parallelepipedAll)] = uI[getIndex(i, j, k, parallelepiped)];
                }
            }
        }
    }

    return uAll;
}
#pragma endregion

#pragma region Results fillers

// Заполнение начальных условий
void fillVectorByInitialValues(vector<vector<double>> &u, const SolverVariables &variables) {
    fillBoundaryValues(u[0], 0, variables);
    fillBoundaryValues(u[1], variables.tau, variables);

    ProcessParallelepiped target = variables.processParallelepiped;
    int N = variables.N;

    int xMin = max(target.xMin, 1);
    int xMax = max(target.xMax, N - 1);

    int yMin = max(target.yMin, 1);
    int yMax = max(target.yMax, N - 1);

    int zMin = max(target.zMin, 1);
    int zMax = max(target.zMax, N - 1);

    GridSteps H = variables.H;

#pragma omp parallel for collapse(3)
    for (int i = xMin; i <= xMax; i++) {
        for (int j = yMin; j <= yMax; j++) {
            for (int k = zMin; k < zMax; k++) {
                u[0][getLocalIndex(i, j, k, variables)] = getPhi(i * H.x, j * H.y, k * H.z, variables.L);
            }
        }
    }

    vector<vector<double>> recv = sendRecvValues(u[0], variables);

#pragma omp parallel for collapse(3)
    for (int i = xMin; i <= xMax; i++) {
        for (int j = yMin; j <= yMax; j++) {
            for (int k = zMin; k <= zMax; k++) {
                u[1][getLocalIndex(i, j, k, variables)] =
                        u[0][getLocalIndex(i, j, k, variables)] +
                        variables.tau * variables.tau / 2
                        * calculateLaplaceOperator(u[0], i, j, k, recv, H, variables);
            }
        }
    }
}

void fillNextLayer(const vector<double> &u0, const vector<double> &u1, vector<double> u, double t,
                   const SolverVariables &variables) {
    int xMin = max(variables.processParallelepiped.xMin, 1);
    int xMax = max(variables.processParallelepiped.xMax, variables.N - 1);

    int yMin = max(variables.processParallelepiped.yMin, 1);
    int yMax = max(variables.processParallelepiped.yMax, variables.N - 1);

    int zMin = max(variables.processParallelepiped.zMin, 1);
    int zMax = max(variables.processParallelepiped.zMax, variables.N - 1);

    vector<vector<double>> uRecv = sendRecvValues(u1, variables);

#pragma omp parallel for collapse(3)
    for (int i = xMin; i <= xMax; i++) {
        for (int j = yMin; j <= yMax; j++) {
            for (int k = zMin; k <= zMax; k++) {
                u[getLocalIndex(i, j, k, variables)] =
                        2 * u1[getLocalIndex(i, j, k, variables)] -
                        u0[getLocalIndex(i, j, k, variables)] + variables.tau * variables.tau *
                                                                calculateLaplaceOperator(u1, i, j, k,
                                                                                         uRecv,
                                                                                         variables.H,
                                                                                         variables);
            }
        }
    }

    fillBoundaryValues(u, t, variables);
}

void fillAnalyticalValues(vector<double> u, double t, const SolverVariables &variables) {
    ProcessParallelepiped parallelepiped = variables.processParallelepiped;
    GridSteps H = variables.H;
#pragma omp parallel for collapse(3)
    for (int i = parallelepiped.xMin; i <= parallelepiped.xMax; i++) {
        for (int j = parallelepiped.yMin; j <= parallelepiped.yMax; j++) {
            for (int k = parallelepiped.zMin; k <= parallelepiped.zMax; k++) {
                u[getLocalIndex(i, j, k, variables)] =
                        getAnalyticValue(i * H.x, j * H.y, k * H.z, t, variables.L);
            }
        }
    }
}

void fillDifferenceValues(vector<double> u, double t, const SolverVariables &variables) {
    ProcessParallelepiped parallelepiped = variables.processParallelepiped;
    GridSteps H = variables.H;
#pragma omp parallel for collapse(3)
    for (int i = parallelepiped.xMin; i <= parallelepiped.xMax; i++) {
        for (int j = parallelepiped.yMin; j <= parallelepiped.yMax; j++) {
            for (int k = parallelepiped.zMin; k <= parallelepiped.zMax; k++) {
                u[getLocalIndex(i, j, k, variables)] =
                        fabs(u[getLocalIndex(i, j, k, variables)]) -
                        getAnalyticValue(i * H.x, j * H.y, k * H.z, t, variables.L);
            }
        }
    }
}

// Оценка погрешности на слое
double evaluateError(vector<double> u, double t, const SolverVariables &variables) {
    double localError = 0, error = 0;
    int N = variables.N;
    GridSteps H = variables.H;
    ProcessParallelepiped parallelepiped = variables.processParallelepiped;

#pragma omp parallel for collapse(3) reduction(max: localError)
    for (int i = parallelepiped.xMin; i <= parallelepiped.xMax; i++) {
        for (int j = parallelepiped.yMin; j <= parallelepiped.yMax; j++) {
            for (int k = parallelepiped.zMin; k <= parallelepiped.zMax; k++) {
                localError = max(localError, fabs(u[getLocalIndex(i, j, k, variables)] -
                                                  getAnalyticValue(i * H.x, j * H.y, k * H.z, t, variables.L)));
            }
        }
    }

    MPI_Reduce(&localError, &error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return error;
}

// сохранение слоя в формате json. Для построения графика на спец ресурсе
void
saveValues(vector<double> u, double t, const vector<ProcessParallelepiped> &parallelepipeds, const char *filename,
           const SolverVariables &variables) {
    vector<double> uAll = sendRecvTotal(u, parallelepipeds, variables);

    if (variables.processId != 0) {
        return;
    }

    ofstream f(filename);

    f << "{" << endl;
    f << "    \"Lx\": " << variables.L.x << ", " << endl;
    f << "    \"Ly\": " << variables.L.y << ", " << endl;
    f << "    \"Lz\": " << variables.L.z << ", " << endl;
    f << "    \"N\": " << variables.N << ", " << endl;
    f << "    \"t\": " << t << ", " << endl;
    f << "    \"u\": [" << endl;

    bool wasPrinted = false;

    for (int i = 0; i < variables.layerSize; i++) {
        if (wasPrinted) {
            f << ", " << endl;
        } else {
            wasPrinted = true;
        }

        f << "    " << uAll[i];
    }

    f << endl;
    f << "    ]" << endl;
    f << "}" << endl;

    f.close();
}

#pragma endregion

double makeSolution(SolverVariables &variables) {
    int layerSize = variables.layerSize;
    int steps = variables.steps;
    int N = variables.N;
    double tau = variables.tau;

    // Разделяем параллелепипед по процессам на sub-параллелепипеды
    vector<ProcessParallelepiped> parallelepipeds = splitParallelepiped(N, variables.countOfProcesses);
    variables.processParallelepiped = parallelepipeds[variables.processId];

    fillNeighbours(parallelepipeds, variables);

    vector<vector<double>> u{vector<double>(layerSize), vector<double>(layerSize), vector<double>(layerSize)};

    fillVectorByInitialValues(u, variables);

    double error0 = evaluateError(u[0], 0, variables);
    double error1 = evaluateError(u[1], variables.tau, variables);

    if (variables.processId == 0) {
        cout << "Layer 0 max error: " << error0 << endl;
        cout << "Layer 1 max error: " << error1 << endl;
    }


    double t = steps * tau;
    for (int step = 2; step <= steps; step++) {
        fillNextLayer(u[(step + 1) % 3], u[(step + 2) % 3], u[step % 3], step * variables.tau, variables);

        double error = evaluateError(u[steps % 3], t, variables);
        if (variables.processId == 0) {
            cout << "Layer " << step << " max error: " << error << endl;
        }
    }

    saveValues(u[steps % 3], t, parallelepipeds, "numerical.json", variables);

    fillDifferenceValues(u[steps % 3], t, variables);
    saveValues(u[steps % 3], t, parallelepipeds, "difference.json", variables);

    fillAnalyticalValues(u[0], t, variables);
    saveValues(u[0], t, parallelepipeds, "analytical.json", variables);

    return evaluateError(u[steps % 3], t, variables);
}

// ЛАЗАРЕВ В.А. / 628 группа / 2 вариант
int main(int argc, char *argv[]) {
    SolverVariables variables{};

    int processId, countOfProcesses;
    // Инициализация MPI, создание группы процессов, создание области связи MPI_COMM_WORLD
    MPI_Init(nullptr, nullptr);
    // Определяем номер процесса (сохранится в переменную processId)
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    // Определение числа процессов в области связи MPI_COMM_WORLD (сохранится в переменную countOfProcesses)
    MPI_Comm_size(MPI_COMM_WORLD, &countOfProcesses);

    double start = MPI_Wtime();
    initVariables(argc, argv, variables, processId, countOfProcesses);

    // Устанавливаем количество omp-потоков
    omp_set_num_threads(variables.ompThreadsCount);

    double error = makeSolution(variables);

    double end = MPI_Wtime();
    double diffTime = end - start;

    double minTime, maxTime, avgTime;

    // Отбираем минимальный diffTime по всем процессам
    MPI_Reduce(&diffTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, MAIN_PROCESS_ID, MPI_COMM_WORLD);
    // Отбираем максимальный diffTime по всем процессам
    MPI_Reduce(&diffTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, MAIN_PROCESS_ID, MPI_COMM_WORLD);
    // Отбираем средний diffTime по всем процессам
    MPI_Reduce(&diffTime, &avgTime, 1, MPI_DOUBLE, MPI_SUM, MAIN_PROCESS_ID, MPI_COMM_WORLD);

    avgTime = avgTime / countOfProcesses;

    // Чтоб печатал информацию только один процесс
    if (processId == MAIN_PROCESS_ID) {
        cout << "OMP threads: " << variables.ompThreadsCount << endl;
        cout << "Final error: " << error << endl;
        cout << "Minimal time (s): " << minTime << endl;
        cout << "Maximum time (s): " << maxTime << endl;
        cout << "Average time (s): " << avgTime << endl << endl;
    }

    MPI_Finalize();

    return 0;
}