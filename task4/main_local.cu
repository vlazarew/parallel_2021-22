#include <iostream>
#include <fstream>
#include <iomanip>
#include <mpi/mpi.h>
//#include <mpi.h>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>

using namespace std;

#pragma region Entities and constants
const int MAIN_PROCESS_ID = 0;
//const bool USE_GPU = true;
// Размер GPU блока
const int GPU_BLOCK_SIZE = 512;

// Оси разбиения
enum Axis {
    X,
    Y,
    Z,
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

    // Id процесса / количество MPI-процессов
    int processId, countOfProcesses;

    // Параллелпипед, обрабтываемый отдельным процессом
    ProcessParallelepiped processParallelepiped;
    // Параллелепипеды-соседи на передачу
    vector <ProcessParallelepiped> send;
    // Параллелепипеды-соседи на прием
    vector <ProcessParallelepiped> recv;
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
            }
            catch (...) {
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
            }
            catch (...) {
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
    variables.N = getIntValueFromArg("-N=", argc, argv, 5);
    variables.K = getIntValueFromArg("-K=", argc, argv, 50);
    variables.steps = getIntValueFromArg("-steps=", argc, argv, 20);
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

// Конструктор параллелепипеда по макс/мин координатам
ProcessParallelepiped createParallelepiped(int xMin, int xMax, int yMin, int yMax, int zMin, int zMax) {
    int dx = xMax - xMin + 1;
    int dy = yMax - yMin + 1;
    int dz = zMax - zMin + 1;
    int size = dx * dy * dz;

    return ProcessParallelepiped{xMin, xMax, yMin, yMax, zMin, zMax, dx, dy, dz, size};
}

// Блочное разбиение параллелепипеда
void makeSplit(int xMin, int xMax, int yMin, int yMax, int zMin, int zMax, int countOfProcesses, Axis axis,
               vector <ProcessParallelepiped> &subParallelepipeds) {

    // Если один процесс, то ему достается полный параллелепипед
    if (countOfProcesses == 1) {
        subParallelepipeds.push_back(createParallelepiped(xMin, xMax, yMin, yMax, zMin, zMax));
        return;
    }

    // Делаем область по текущей оси, делаем паралл, разбиваем дальше
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

    // Для оси делим область пополам и запускаем рекурсию для sub-параллелепипедов
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

// Запуск разбиения параллелпипеда на более мелкие
vector <ProcessParallelepiped> splitParallelepiped(int N, int countOfProcesses) {
    vector <ProcessParallelepiped> subParallelepipeds;

    // Коэффициенты - параметры исходного параллелепипеда
    makeSplit(0, N, 0, N, 0, N, countOfProcesses, X, subParallelepipeds);

    return subParallelepipeds;
}

#pragma endregion

#pragma region Actions with neighbours

// Проверка на то, что первый параллелепипед (координаты с 1) находится внутри второго (координаты с 2)
bool isInside(int xMin1, int xMax1, int yMin1, int yMax1, int xMin2, int xMax2, int yMin2, int yMax2) {
    return xMin2 <= xMin1 && xMax1 <= xMax2 && yMin2 <= yMin1 && yMax1 <= yMax2;
}

// Получить соприкосновение двух параллелипедов (true - если соприкосновение есть, иначе - false). result - соприкосновение
bool
getNeighbours(const ProcessParallelepiped &first, const ProcessParallelepiped &second, ProcessParallelepiped &result) {
    // Если начало одного == конец второго по оси X
    if (first.xMin == second.xMax + 1 || second.xMin == first.xMax + 1) {
        // Берем точку соприкосновения двух параллелепипедов
        int x = (first.xMin == second.xMax + 1) ? first.xMin : first.xMax;

        // Если fisrt внутри second
        if (isInside(first.yMin, first.yMax, first.zMin, first.zMax, second.yMin, second.yMax, second.zMin,
                     second.zMax)) {
            // Создадим параллепипед по X размером 1 (соприкосновение блоков) и размерами first => result - параллелепипед, от которого получать/посылать информацию
            result = createParallelepiped(x, x, first.yMin, first.yMax, first.zMin, first.zMax);
            return true;
        }

        // Если second внутри first
        if (isInside(second.yMin, second.yMax, second.zMin, second.zMax, first.yMin, first.yMax, first.zMin,
                     first.zMax)) {
            // Создадим параллепипед по X размером 1 (соприкосновение блоков) и размерами second => result - параллелепипед, от которого получать/посылать информацию
            result = createParallelepiped(x, x, second.yMin, second.yMax, second.zMin, second.zMax);
            return true;
        }

        return false;
    }

    // Если начало одного == конец второго по оси Y
    if (first.yMin == second.yMax + 1 || second.yMin == first.yMax + 1) {
        // Берем точку соприкосновения двух параллелепипедов
        int y = (first.yMin == second.yMax + 1) ? first.yMin : first.yMax;

        // Если fisrt внутри second
        if (isInside(first.xMin, first.xMax, first.zMin, first.zMax, second.xMin, second.xMax, second.zMin,
                     second.zMax)) {
            // Создадим параллепипед по Y размером 1 (соприкосновение блоков) и размерами first => result - параллелепипед, от которого получать/посылать информацию
            result = createParallelepiped(first.xMin, first.xMax, y, y, first.zMin, first.zMax);
            return true;
        }

        // Если second внутри first
        if (isInside(second.xMin, second.xMax, second.zMin, second.zMax, first.xMin, first.xMax, first.zMin,
                     first.zMax)) {
            // Создадим параллепипед по Y размером 1 (соприкосновение блоков) и размерами second => result - параллелепипед, от которого получать/посылать информацию
            result = createParallelepiped(second.xMin, second.xMax, y, y, second.zMin, second.zMax);
            return true;
        }

        return false;
    }

    // Если начало одного == конец второго по оси Z
    if (first.zMin == second.zMax + 1 || second.zMin == first.zMax + 1) {
        // Берем точку соприкосновения двух параллелепипедов
        int z = (first.zMin == second.zMax + 1) ? first.zMin : first.zMax;

        // Если fisrt внутри second
        if (isInside(first.xMin, first.xMax, first.yMin, first.yMax, second.xMin, second.xMax, second.yMin,
                     second.yMax)) {
            // Создадим параллепипед по Z размером 1 (соприкосновение блоков) и размерами first => result - параллелепипед, от которого получать/посылать информацию
            result = createParallelepiped(first.xMin, first.xMax, first.yMin, first.yMax, z, z);
            return true;
        }

        // Если second внутри first
        if (isInside(second.xMin, second.xMax, second.yMin, second.yMax, first.xMin, first.xMax, first.yMin,
                     first.yMax)) {
            // Создадим параллепипед по Z размером 1 (соприкосновение блоков) и размерами second => result - параллелепипед, от которого получать/посылать информацию
            result = createParallelepiped(second.xMin, second.xMax, second.yMin, second.yMax, z, z);
            return true;
        }

        return false;
    }

    return false;
}

// Заполняем соседей
void fillNeighbours(const vector <ProcessParallelepiped> &parallelepipeds, vector<int> &processIds,
                    vector <ProcessParallelepiped> &send,
                    vector <ProcessParallelepiped> &recv, int processId, int countOfProcesses,
                    ProcessParallelepiped target) {
    send.clear();
    recv.clear();
    processIds.clear();

    for (int i = 0; i < countOfProcesses; i++) {
        // Самого себя не обрабатываем
        if (i == processId) {
            continue;
        }

        ProcessParallelepiped sendParallelepiped;
        ProcessParallelepiped recvParallelepiped;

        ProcessParallelepiped processParallelepiped = parallelepipeds[i];
        // Получаем соседей, которым будем отправлять информацию
        if (!getNeighbours(target, processParallelepiped, sendParallelepiped)) {
            continue;
        }

        // Получаем соседей, от которых будем получать информацию
        getNeighbours(processParallelepiped, target, recvParallelepiped);
        processIds.push_back(i);
        send.push_back(sendParallelepiped);
        recv.push_back(recvParallelepiped);
    }
}

#pragma endregion

#pragma region Math calculations

// Аналитическое решение
__host__ __device__ double getAnalyticValue(double x, double y, double z, double t, Parallelepiped L) {
    double at = M_PI * sqrt(1 / pow(L.x, 2) + 1 / pow(L.y, 2) + 4 / pow(L.z, 2));

    return sin(M_PI * x / L.x) * sin(M_PI * y / L.y) * sin(2 * z * M_PI / L.z) * cos(at * t + 2 * M_PI);
}

// Начальные условия
__host__ __device__ double getPhi(double x, double y, double z, Parallelepiped L) {
    return getAnalyticValue(x, y, z, 0, L);
}

// Получить индекс по x, y, z для конкрентного параллелепипеда
__host__ __device__ int getIndex(int x, int y, int z, ProcessParallelepiped target) {
    return (x - target.xMin) * target.dy * target.dz + (y - target.yMin) * target.dz + (z - target.zMin);
}

// Получить локальный индекс
__host__ __device__ int getLocalIndex(int x, int y, int z, SolverVariables variables) {
    return getIndex(x, y, z, variables.processParallelepiped);
}

__host__ __device__ double findValue(int size, const double *u, int x, int y, int z, const double *recv,
                                     const ProcessParallelepiped *recvNeighboursData,
                                     ProcessParallelepiped target) {
    if (target.xMin <= x && target.xMax >= x &&
        target.yMin <= y && target.yMax >= y &&
        target.zMin <= z && target.zMax >= z) {
        return u[getIndex(x, y, z, target)];
    }

    int offset = 0;

    for (int index = 0; index < size; index++) {
        ProcessParallelepiped parallelepiped = recvNeighboursData[index];

        if (x < parallelepiped.xMin || x > parallelepiped.xMax ||
            y < parallelepiped.yMin || y > parallelepiped.yMax ||
            z < parallelepiped.zMin || z > parallelepiped.zMax) {
            offset += recvNeighboursData[index].size;
            continue;
        }

        return recv[offset + getIndex(x, y, z, parallelepiped)];
    }

    return -1;
}

// Оператор Лапласа
__host__ __device__ double
calculateLaplaceOperator(int recvSize, const double *uData, int x, int y, int z, const double *uRecvData,
                         const ProcessParallelepiped *recvNeighboursData, ProcessParallelepiped target, double hx,
                         double hy, double hz) {
    double coeff = 2 * uData[getIndex(x, y, z, target)];

    double dx = (findValue(recvSize, uData, x - 1, y, z, uRecvData, recvNeighboursData, target) +
                 findValue(recvSize, uData, x + 1, y, z, uRecvData, recvNeighboursData, target) - coeff) /
                (hx * hx);
    double dy = (findValue(recvSize, uData, x, y - 1, z, uRecvData, recvNeighboursData, target) +
                 findValue(recvSize, uData, x, y + 1, z, uRecvData, recvNeighboursData, target) - coeff) /
                (hy * hy);
    double dz = (findValue(recvSize, uData, x, y, z - 1, uRecvData, recvNeighboursData, target) +
                 findValue(recvSize, uData, x, y, z + 1, uRecvData, recvNeighboursData, target) - coeff) /
                (hz * hz);

    return dx + dy + dz;
}

// Получить значение граничного условия
__host__ __device__ double
getBoundaryValue(int x, int y, int z, double t, double hx, double hy, double hz, Parallelepiped L, int N) {
    // По X - первого рода
    if (x == 0 || x == N) {
        return 0;
    }

    // По Y - первого рода
    if (y == 0 || y == N) {
        return 0;
    }

    // По Z - периодическое значение
    if (z == 0 || z == N) {
        return getAnalyticValue(x * hx, y * hy, z * hz, t, L);
    }

    return 0;
}

__global__ void
fillBoundaryValuesX(double *u, ProcessParallelepiped target, int x, double hx, double hy, double hz, double tau,
                    Parallelepiped L, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= target.dy * target.dz) {
        return;
    }

    int y = target.yMin + index / target.dz;
    int z = target.zMin + index % target.dz;

    u[getIndex(x, y, z, target)] = getBoundaryValue(x, y, z, tau, hx, hy, hz, L, N);
}

__global__ void
fillBoundaryValuesY(double *u, ProcessParallelepiped target, int y, double hx, double hy, double hz, double tau,
                    Parallelepiped L, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= target.dx * target.dz) {
        return;
    }

    int x = target.xMin + index / target.dz;
    int z = target.zMin + index % target.dz;

    u[getIndex(x, y, z, target)] = getBoundaryValue(x, y, z, tau, hx, hy, hz, L, N);
}

__global__ void
fillBoundaryValuesZ(double *u, ProcessParallelepiped target, int z, double hx, double hy, double hz, double tau,
                    Parallelepiped L, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= target.dx * target.dy) {
        return;
    }

    int x = target.xMin + index / target.dy;
    int y = target.yMin + index % target.dy;

    u[getIndex(x, y, z, target)] = getBoundaryValue(x, y, z, tau, hx, hy, hz, L, N);
}

//void fillBoundaryValuesDefault(vector<double> &u, double tau, const SolverVariables &variables) {
//    int N = variables.N;
//
//    ProcessParallelepiped processParallelepiped = variables.processParallelepiped;
//
//    // Граница при 0
//    if (processParallelepiped.xMin == 0) {
//        for (int y = processParallelepiped.yMin; y <= processParallelepiped.yMax; y++) {
//            for (int z = processParallelepiped.zMin; z <= processParallelepiped.zMax; z++) {
//                // u[индекс по x, y, z для текущего параллелепипеда]
//                u[getLocalIndex(processParallelepiped.xMin, y, z, variables)] =
//                        getBoundaryValue(processParallelepiped.xMin, y, z, tau, variables);
//            }
//        }
//    }
//
//    // Граница при N
//    if (processParallelepiped.xMax == N) {
//        for (int y = processParallelepiped.yMin; y <= processParallelepiped.yMax; y++) {
//            for (int z = processParallelepiped.zMin; z <= processParallelepiped.zMax; z++) {
//                u[getLocalIndex(processParallelepiped.xMax, y, z, variables)] =
//                        getBoundaryValue(processParallelepiped.xMax, y, z, tau, variables);
//            }
//        }
//    }
//
//    // Граница при 0
//    if (processParallelepiped.yMin == 0) {
//        for (int x = processParallelepiped.xMin; x <= processParallelepiped.xMax; x++) {
//            for (int z = processParallelepiped.zMin; z <= processParallelepiped.zMax; z++) {
//                u[getLocalIndex(x, processParallelepiped.yMin, z, variables)] =
//                        getBoundaryValue(x, processParallelepiped.yMin, z, tau, variables);
//            }
//        }
//    }
//
//    // Граница при N
//    if (processParallelepiped.yMax == N) {
//        for (int x = processParallelepiped.xMin; x <= processParallelepiped.xMax; x++) {
//            for (int z = processParallelepiped.zMin; z <= processParallelepiped.zMax; z++) {
//                u[getLocalIndex(x, processParallelepiped.yMax, z, variables)] =
//                        getBoundaryValue(x, processParallelepiped.yMax, z, tau, variables);
//            }
//        }
//    }
//
//    // Граница при 0
//    if (processParallelepiped.zMin == 0) {
//        for (int x = processParallelepiped.xMin; x <= processParallelepiped.xMax; x++) {
//            for (int y = processParallelepiped.yMin; y <= processParallelepiped.yMax; y++) {
//                u[getLocalIndex(x, y, processParallelepiped.zMin, variables)] =
//                        getBoundaryValue(x, y, processParallelepiped.zMin, tau, variables);
//            }
//        }
//    }
//
//    // Граница при N
//    if (processParallelepiped.zMax == N) {
//        for (int x = processParallelepiped.xMin; x <= processParallelepiped.xMax; x++)
//            for (int y = processParallelepiped.yMin; y <= processParallelepiped.yMax; y++)
//                u[getLocalIndex(x, y, processParallelepiped.zMax, variables)] =
//                        getBoundaryValue(x, y, processParallelepiped.zMax, tau, variables);
//    }
//}

__global__ void
fillZeroLayerKernel(double *u0, int total, int xMin, int yMin, int zMin, int dy, int dz, ProcessParallelepiped target,
                    double hx, double hy, double hz, Parallelepiped L) {
    int index = blockIdx.x * blockDim.x * threadIdx.x;

    if (index >= total) {
        return;
    }

    int x = xMin + index / (dy * dz);
    int y = yMin + index % (dy * dz) / dz;
    int z = zMin + index % dz;

    u0[getIndex(x, y, z, target)] = getPhi(x * hx, y * hy, z * hz, L);
}

__global__ void
fillFirstLayerKernel(double *u1, const double *u0, const double *u0Recv, const ProcessParallelepiped *recv, int total,
                     int size, int xMin, int yMin, int zMin, int dy, int dz, ProcessParallelepiped target, double hx,
                     double hy, double hz, double tau) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= total) {
        return;
    }

    int x = xMin + index / (dy * dz);
    int y = yMin + index % (dy * dz) / dz;
    int z = zMin + index % dz;

    u1[getIndex(x, y, z, target)] = u0[getIndex(x, y, z, target)] + tau * tau / 2 *
                                                                    calculateLaplaceOperator(size, u0, x, y, z, u0Recv,
                                                                                             recv, target, hx, hy, hz);
}

__global__ void fillNextLayerKernel(double *u, const double *u0, const double *u1, const double *uRecv,
                                    const ProcessParallelepiped *recvNeighbours, int total, int size, int xMin,
                                    int yMin, int zMin, int dy,
                                    int dz, ProcessParallelepiped target, double hx, double hy, double hz, double tau) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index > total) {
        return;
    }

    int x = xMin + index / (dy * dz);
    int y = yMin + index % (dy * dz) / dz;
    int z = zMin + index % dz;

    u[getIndex(x, y, z, target)] = 2 * u1[getIndex(x, y, z, target)] - u0[getIndex(x, y, z, target)] +
                                   tau * tau *
                                   calculateLaplaceOperator(size, u1, x, y, z, uRecv, recvNeighbours, target, hx, hy,
                                                            hz);
}

template<typename T>
inline T *vectorToDevice(const vector <T> &vector, bool fill = true) {
    T *deviceVector;
    cudaMalloc((void **) &deviceVector, vector.size() * sizeof(T));

    if (fill) {
        cudaMemcpyAsync(deviceVector, vector.data(), vector.size() * sizeof(T), cudaMemcpyHostToDevice, 0);
    }

    return deviceVector;
}

void fillBoundaryValuesKernel(vector<double> &u, double tau, double hx, double hy, double hz, Parallelepiped L,
                              ProcessParallelepiped target, int N) {
    double *uDevice = vectorToDevice(u);

    if (target.xMin == 0) {
        fillBoundaryValuesX<<<((target.dy * target.dz + GPU_BLOCK_SIZE - 1) /
                               GPU_BLOCK_SIZE), GPU_BLOCK_SIZE>>>(uDevice, target, 0, hx, hy, hz, tau, L, N);
    }
    if (target.xMax == N) {
        fillBoundaryValuesX<<<((target.dy * target.dz + GPU_BLOCK_SIZE - 1) /
                               GPU_BLOCK_SIZE), GPU_BLOCK_SIZE>>>(uDevice, target, N, hx, hy, hz, tau, L, N);
    }
    if (target.yMin == 0) {
        fillBoundaryValuesY<<<((target.dx * target.dz + GPU_BLOCK_SIZE - 1) /
                               GPU_BLOCK_SIZE), GPU_BLOCK_SIZE>>>(uDevice, target, 0, hx, hy, hz, tau, L, N);
    }
    if (target.yMax == N) {
        fillBoundaryValuesY<<<((target.dx * target.dz + GPU_BLOCK_SIZE - 1) /
                               GPU_BLOCK_SIZE), GPU_BLOCK_SIZE>>>(uDevice, target, N, hx, hy, hz, tau, L, N);
    }
    if (target.zMin == 0) {
        fillBoundaryValuesZ<<<((target.dx * target.dy + GPU_BLOCK_SIZE - 1) /
                               GPU_BLOCK_SIZE), GPU_BLOCK_SIZE>>>(uDevice, target, 0, hx, hy, hz, tau, L, N);
    }
    if (target.zMax == N) {
        fillBoundaryValuesZ<<<((target.dx * target.dy + GPU_BLOCK_SIZE - 1) /
                               GPU_BLOCK_SIZE), GPU_BLOCK_SIZE>>>(uDevice, target, N, hx, hy, hz, tau, L, N);
    }

    cudaMemcpyAsync(u.data(), uDevice, u.size() * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaFree(uDevice);
}

// Заполнение граничными значениями
void fillBoundaryValues(vector<double> &u, double tau, double hx, double hy, double hz, Parallelepiped L,
                        ProcessParallelepiped target, int N) {
    if (target.xMin > 0 && target.xMax < N &&
        target.yMin > 0 && target.yMax < N &&
        target.zMin > 0 && target.zMax < N) {
        return;
    }

//    if (!USE_GPU) {
//        fillBoundaryValuesDefault(u, tau, variables);
//    } else {
    fillBoundaryValuesKernel(u, tau, hx, hy, hz, L, target, N);
//    }
}

#pragma endregion

#pragma region data send recv

// Собрать параллелепипед в обособленный массив (вектор)
vector<double>
packParallelepiped(const vector<double> &u, const ProcessParallelepiped &parallelepiped,
                   ProcessParallelepiped target) {
    vector<double> packed(parallelepiped.size);

    for (int i = parallelepiped.xMin; i <= parallelepiped.xMax; i++) {
        for (int j = parallelepiped.yMin; j <= parallelepiped.yMax; j++) {
            for (int k = parallelepiped.zMin; k <= parallelepiped.zMax; k++) {
                packed[getIndex(i, j, k, parallelepiped)] = u[getIndex(i, j, k, target)];
            }
        }
    }

    return packed;
}

// Отправка и получение соседних значений
vector<double> sendRecvValues(const vector<double> &u, vector<int> processIds, vector <ProcessParallelepiped> send,
                              vector <ProcessParallelepiped> recv, ProcessParallelepiped target) {
    vector<double> uRecv;
    int offset = 0;

    for (int i = 0; i < processIds.size(); i++) {
        vector<double> packed = packParallelepiped(u, send[i], target);
        uRecv.insert(uRecv.end(), recv[i].size, 0);

        vector <MPI_Request> requests(2);
        vector <MPI_Status> statuses(2);

        // Отправляем буффер с начальным адресом packed.data() и кол-вом элементов variables.send[i].size типа double
        // в процесс с id variables.processIds[i] с пометкой 0 в коммутаторе MPI_COMM_WORLD. Выходное значение записывается в requests[0]
        MPI_Isend(packed.data(), send[i].size, MPI_DOUBLE, processIds[i], 0, MPI_COMM_WORLD, &requests[0]);
        // Получаем буффер с начальным адресом recv[i].data() и кол-вом элементов variables.recv[i].size типа double
        // из процесса с id variables.processIds[i] с пометкой 0 в коммутаторе MPI_COMM_WORLD. Выходное значение записывается в requests[1]
        MPI_Irecv(uRecv.data() + offset, recv[i].size, MPI_DOUBLE, processIds[i], 0, MPI_COMM_WORLD, &requests[1]);
        // 2 = размер списка requests. Полученные статусы всех Isend/Irecv записываются в statuses
        // Ожидает выполнения всех MPI_Requests из списка requests
        MPI_Waitall(2, requests.data(), statuses.data());
        offset += recv[i].size;
    }

    return uRecv;
}

// Отправка и получение общих значений
vector<double>
sendRecvTotal(const vector<double> &u, const vector <ProcessParallelepiped> &parallelepipeds,
              const SolverVariables &variables) {
    if (variables.processId != MAIN_PROCESS_ID) {
        MPI_Request request;
        MPI_Status status;

        // Отправляем буффер с начальным адресом u.data() и кол-вом элементов variables.processParallelepiped.size типа double
        // в процесс с id MAIN_PROCESS_ID == 0 с пометкой 0 в коммутаторе MPI_COMM_WORLD. Выходное значение записывается в request
        MPI_Isend(u.data(), variables.processParallelepiped.size, MPI_DOUBLE, MAIN_PROCESS_ID, 0, MPI_COMM_WORLD,
                  &request);
        // 1 = кол-во requests. Полученный статус всех Isend записывается в status
        // Ожидает выполнения всех MPI_Requests из списка requests
        MPI_Waitall(1, &request, &status);
        return u;
    }

    vector<double> uAll(variables.layerSize);
    ProcessParallelepiped parallelepipedAll = createParallelepiped(0, variables.N, 0, variables.N, 0, variables.N);

    for (int index = 0; index < variables.countOfProcesses; index++) {
        ProcessParallelepiped parallelepiped = parallelepipeds[index];
        vector<double> uI(parallelepiped.size);

        if (index == variables.processId) {
            uI = u;
        } else {
            vector <MPI_Request> requests(1);
            vector <MPI_Status> statuses(1);

            // Получаем буффер с начальным адресом uI.data() и кол-вом элементов parallelepiped.size типа double
            // из процесса с id index с пометкой 0 в коммутаторе MPI_COMM_WORLD. Выходное значение записывается в requests[0]
            MPI_Irecv(uI.data(), parallelepiped.size, MPI_DOUBLE, index, 0, MPI_COMM_WORLD, &requests[0]);
            // 1 = кол-во requests. Полученный статус всех Irecv записывается в statuses
            // Ожидает выполнения всех MPI_Requests из списка requests
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

//void fillVectorByInitialValuesDefault(vector <vector<double>> &u, const SolverVariables &variables, int xMin, int xMax,
//                                      int yMin, int yMax, int zMin, int zMax) {
//    ProcessParallelepiped processParallelepiped = variables.processParallelepiped;
//    GridSteps H = variables.H;
//    for (int index = 0; index < processParallelepiped.size; index++) {
//        int x = xMin + index / (processParallelepiped.dy * processParallelepiped.dz);
//        int y = yMin + index % (processParallelepiped.dy * processParallelepiped.dz) / processParallelepiped.dz;
//        int z = zMin + index % processParallelepiped.dz;
//
//        u[0][getLocalIndex(x, y, z, variables)] = getPhi(x * H.x, y * H.y, z * H.z, variables.L);
//    }
//
//    vector<double> recv = sendRecvValues(u[0], variables);
//
//    double coeff = variables.tau * variables.tau / 2;
//    for (int index = 0; index < processParallelepiped.size; index++) {
//        int x = xMin + index / (processParallelepiped.dy * processParallelepiped.dz);
//        int y = yMin + index % (processParallelepiped.dy * processParallelepiped.dz) / processParallelepiped.dz;
//        int z = zMin + index % processParallelepiped.dz;
//
//        u[1][getLocalIndex(x, y, z, variables)] = u[0][getLocalIndex(x, y, z, variables)] + coeff *
//                                                                                            calculateLaplaceOperator(
//                                                                                                    variables.recv.size(),
//                                                                                                    u[0].data(), x, y,
//                                                                                                    z, recv.data(),
//                                                                                                    variables.recv.data(),
//                                                                                                    variables);
//    }
//}

//void fillVectorByInitialValuesKernel(vector<double> &u0, vector<double> &u1, ProcessParallelepiped target, int total,
//                                     int xMin, int xMax, int yMin, int yMax, int zMin, int zMax, double hx, double hy,
//                                     double hz, double dx, double dy, double dz, double tau, Parallelepiped L,
//                                     const vector<int> &processIds,
//                                     const vector <ProcessParallelepiped> &send,
//                                     const vector <ProcessParallelepiped> &recv) {
//
//}

// Заполнение начальных условий
void fillVectorByInitialValues(vector<double> &u0, vector<double> &u1, double hx, double hy, double hz, double tau,
                               Parallelepiped L, int N, const vector<int> &processIds,
                               const vector <ProcessParallelepiped> &send,
                               const vector <ProcessParallelepiped> &recv, ProcessParallelepiped target) {
// Заполняем граничные условия;
    fillBoundaryValues(u0, 0, hx, hy, hz, L, target, N);
    fillBoundaryValues(u1, tau, hx, hy, hz, L, target, N);

    int xMin = max(target.xMin, 1);
    int xMax = min(target.xMax, N - 1);

    int yMin = max(target.yMin, 1);
    int yMax = min(target.yMax, N - 1);

    int zMin = max(target.zMin, 1);
    int zMax = min(target.zMax, N - 1);

    int dx = xMax - xMin + 1;
    int dy = yMax - yMin + 1;
    int dz = zMax - zMin + 1;
    int total = dx * dy * dz;

//    if (!USE_GPU) {
//        fillVectorByInitialValuesDefault(u, variables, xMin, xMax, yMin, yMax, zMin, zMax
//        );
//    } else {
//    fillVectorByInitialValuesKernel(u0, u1, target, total, xMin, xMax, yMin, yMax, zMin, zMax, hx, hy, hz, dx, dy, dz,
//                                    L);
//}

    int nBlocks = (total + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;

    double *u0Device = vectorToDevice(u0, false);
    fillZeroLayerKernel<<<nBlocks, GPU_BLOCK_SIZE>>>(u0Device, total, xMin, yMin, zMin, dy, dz, target, hx, hy, hz, L);
    cudaMemcpyAsync(u0.data(), u0Device, u0.size() * sizeof(double), cudaMemcpyDeviceToHost, 0);

    vector<double> u0Recv = sendRecvValues(u0, processIds, send, recv, target);

    double *u0RecvDevice = vectorToDevice(u0Recv);
    double *u1Device = vectorToDevice(u1, false);
    ProcessParallelepiped *recvNeighboursDevice = vectorToDevice(recv);

    fillFirstLayerKernel<<<nBlocks, GPU_BLOCK_SIZE>>>(u1Device, u0Device, u0RecvDevice, recvNeighboursDevice,
                                                      total, recv.size(), xMin, yMin, zMin, dy, dz, target, hx, hy, hz,
                                                      tau);
    cudaMemcpyAsync(u1.data(), u1Device, u1.size() * sizeof(double), cudaMemcpyDeviceToHost, 0);

    cudaFree(u0Device);
    cudaFree(u0RecvDevice);
    cudaFree(u1Device);
    cudaFree(recvNeighboursDevice);
}

// Заполнение следующего слоя
void
fillNextLayer(const vector<double> &u0, const vector<double> &u1, vector<double> &u, double t, double hx, double hy,
              double hz, double tau, Parallelepiped L, int N,
              const vector<int> &processIds, const vector <ProcessParallelepiped> &send,
              const vector <ProcessParallelepiped> &recv, ProcessParallelepiped target) {
    int xMin = max(target.xMin, 1);
    int xMax = min(target.xMax, N - 1);

    int yMin = max(target.yMin, 1);
    int yMax = min(target.yMax, N - 1);

    int zMin = max(target.zMin, 1);
    int zMax = min(target.zMax, N - 1);

    int dx = xMax - xMin + 1;
    int dy = yMax - yMin + 1;
    int dz = zMax - zMin + 1;
    int total = dx * dy * dz;

    vector<double> uRecv = sendRecvValues(u1, processIds, send, recv, target);

//    if (!USE_GPU) {
//        for (int index = 0; index < processParallelepiped.size; index++) {
//            int x = processParallelepiped.xMin + index / (processParallelepiped.dy * processParallelepiped.dz);
//            int y = processParallelepiped.yMin +
//                    index % (processParallelepiped.dy * processParallelepiped.dz) / processParallelepiped.dz;
//            int z = processParallelepiped.zMin + index % processParallelepiped.dz;
//
//            u[getLocalIndex(x, y, z, variables)] =
//                    2 * u1[getLocalIndex(x, y, z, variables)] - u0[getLocalIndex(x, y, z, variables)] +
//                    variables.tau * variables.tau *
//                    calculateLaplaceOperator(variables.recv.size(), u1.data(), x, y, z, recv.data(),
//                                             variables.recv.data(), variables);
//        }
//    } else {
    int nBlocks = (total + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;

    double *u0Device = vectorToDevice(u0);
    double *u1Device = vectorToDevice(u1);
    double *uDevice = vectorToDevice(u, false);
    double *uRecvDevice = vectorToDevice(uRecv);
    ProcessParallelepiped *recvNeighborsDevice = vectorToDevice(recv);

    fillNextLayerKernel<<<nBlocks, GPU_BLOCK_SIZE>>>(uDevice, u0Device, u1Device, uRecvDevice, recvNeighborsDevice,
                                                     total, recv.size(), xMin, yMin, zMin, dy, dz, target, hx, hy, hz,
                                                     tau);
    cudaMemcpyAsync(u.data(), uDevice, u.size() * sizeof(double), cudaMemcpyDeviceToHost, 0);

    cudaFree(u0Device);
    cudaFree(u1Device);
    cudaFree(uDevice);
    cudaFree(uRecvDevice);
    cudaFree(recvNeighborsDevice);
//    }

    fillBoundaryValues(u, t, hx, hy, hz, L, target, N);
}

// Заполнить аналитическими значениями
void fillAnalyticalValues(vector<double> &u, double t, const SolverVariables &variables) {
    ProcessParallelepiped parallelepiped = variables.processParallelepiped;
    GridSteps H = variables.H;

    for (int x = parallelepiped.xMin; x <= parallelepiped.xMax; x++) {
        for (int y = parallelepiped.yMin; y <= parallelepiped.yMax; y++) {
            for (int z = parallelepiped.zMin; z <= parallelepiped.zMax; z++) {
                u[getLocalIndex(x, y, z, variables)] =
                        getAnalyticValue(x * H.x, y * H.y, z * H.z, t, variables.L);
            }
        }
    }
}

// Заполнить значениями с diff
void fillDifferenceValues(vector<double> &u, double t, const SolverVariables &variables) {
    ProcessParallelepiped parallelepiped = variables.processParallelepiped;
    GridSteps H = variables.H;

    for (int x = parallelepiped.xMin; x <= parallelepiped.xMax; x++) {
        for (int y = parallelepiped.yMin; y <= parallelepiped.yMax; y++) {
            for (int z = parallelepiped.zMin; z <= parallelepiped.zMax; z++) {
                u[getLocalIndex(x, y, z, variables)] =
                        fabs(u[getLocalIndex(x, y, z, variables)]) -
                        getAnalyticValue(x * H.x, y * H.y, z * H.z, t, variables.L);
            }
        }
    }
}

__global__ void EvaluateErrorKernel(double *u, double hx, double hy, double hz, double t, Parallelepiped L,
                                    ProcessParallelepiped target, double *error) {
    int idx = threadIdx.x;
    double max = 0;

    for (int index = idx; index < target.size; index++) {
        int x = target.xMin + index / (target.dy * target.dz);
        int y = target.yMin + index % (target.dy * target.dz) / target.dz;
        int z = target.zMin + index % target.dz;

        double delta = fabs(u[getIndex(x, y, z, target)] - getAnalyticValue(x * hx, y * hy, z * hz, t, L));

        if (delta > max) {
            max = delta;
        }
    }

    __shared__ double r[GPU_BLOCK_SIZE];

    r[idx] = max;
    __syncthreads();

    for (int size = GPU_BLOCK_SIZE / 2; size > 0; size /= 2) {
        if (idx < size) {
            r[idx] = r[idx] > r[idx + size] ? r[idx] : r[idx + size];
        }

        __syncthreads();
    }

    if (idx == 0) {
        *error = r[0];
    }
}

// Оценка погрешности на слое
double evaluateError(const vector<double> &u, double t, double hx, double hy, double hz, Parallelepiped L,
                     ProcessParallelepiped target) {
    double localError = 0;
    double error;

//    if (!USE_GPU) {
//        for (int x = parallelepiped.xMin; x <= parallelepiped.xMax; x++) {
//            for (int y = parallelepiped.yMin; y <= parallelepiped.yMax; y++) {
//                for (int z = parallelepiped.zMin; z <= parallelepiped.zMax; z++) {
//                    localError = max(localError, fabs(u[getLocalIndex(x, y, z, variables)] -
//                                                      getAnalyticValue(x * H.x, y * H.y, z * H.z, t, variables.L)));
//                }
//            }
//        }
//    } else {
    double *uDevice = vectorToDevice(u);
    double *errorDevice;
    cudaMalloc((void **) &errorDevice, sizeof(double));

    EvaluateErrorKernel<<<1, GPU_BLOCK_SIZE>>>(uDevice, hx, hy, hz, t, L, target, errorDevice);

    cudaMemcpyAsync(&localError, errorDevice, 1 * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaFree(uDevice);
    cudaFree(errorDevice);
//    }

    // Отбираем максимальный localError в переменную error по всем процессам
    MPI_Reduce(&localError, &error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return error;
}

// Cохранение слоя в формате json. Для построения графика на спец ресурсе
void saveValues(const vector<double> &u, double t, const vector <ProcessParallelepiped> &parallelepipeds,
                const char *filename,
                const SolverVariables &variables) {
    vector<double> uAll = sendRecvTotal(u, parallelepipeds, variables);

    // Писать в файл может только мейн процесс (id == 0)
    if (variables.processId != MAIN_PROCESS_ID) {
        return;
    }

    ofstream f(filename);

    // Добавляем параметры запуска
    f << "{" << endl;
    f << "    \"Lx\": " << variables.L.x << ", " << endl;
    f << "    \"Ly\": " << variables.L.y << ", " << endl;
    f << "    \"Lz\": " << variables.L.z << ", " << endl;
    f << "    \"N\": " << variables.N << ", " << endl;
    f << "    \"t\": " << t << ", " << endl;
    f << "    \"u\": [" << endl;

    bool isFirstValuePrinted = false;

    // Печатаем слой
    for (int i = 0; i < variables.layerSize; i++) {
        if (isFirstValuePrinted) {
            f << ", " << endl;
        } else {
            isFirstValuePrinted = true;
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
    int steps = variables.steps;
    GridSteps H = variables.H;
    int N = variables.N;
    double tau = variables.tau;
    int processId = variables.processId;
    int countOfProcesses = variables.countOfProcesses;

    vector<int> processIds;
    vector <ProcessParallelepiped> send;
    vector <ProcessParallelepiped> recv;

    // Разделяем параллелепипед по процессам на sub-параллелепипеды (Step 2)
    vector <ProcessParallelepiped> parallelepipeds = splitParallelepiped(N, variables.countOfProcesses);

    // Запоминаем рабочий параллелепипед
    variables.processParallelepiped = parallelepipeds[variables.processId];
    ProcessParallelepiped target = parallelepipeds[variables.processId];

    // Заполняем соседей
    fillNeighbours(parallelepipeds, processIds, send, recv, processId, countOfProcesses, target);

    vector <vector<double>> u(3, vector<double>(variables.processParallelepiped.size));

    // Заполняем начальнные условия (step 4)
    fillVectorByInitialValues(u[0], u[1], H.x, H.y, H.z, tau, variables.L, N, processIds, send, recv, target);
//    if (variables.processId == MAIN_PROCESS_ID) {
//        cout << "u[0] = ";
//        for (int i = 0; i < u[0].size(); i++) {
//            cout << u[0][i] << ", ";
//        }
//        cout << endl;
//
//        cout << "u[1] = ";
//        for (int i = 0; i < u[1].size(); i++) {
//            cout << u[1][i] << ", ";
//        }
//        cout << endl;
//    }

    double error0 = evaluateError(u[0], 0, H.x, H.y, H.z, variables.L, target);
    double error1 = evaluateError(u[1], tau, H.x, H.y, H.z, variables.L, target);

    if (variables.processId == MAIN_PROCESS_ID) {
        cout << "Layer 0 max error: " << error0 << endl;
        cout << "Layer 1 max error: " << error1 << endl;
    }

    double t = steps * tau;
    for (int step = 2; step <= steps; step++) {
        // Заполняем следующий слой u (step 5 + 6)
        fillNextLayer(u[(step + 1) % 3], u[(step + 2) % 3], u[step % 3], step * tau, H.x, H.y, H.z, tau, variables.L, N,
                      processIds, send, recv, target);

        // Вычисляем максимальную ошибку (step 8)
        double error = evaluateError(u[steps % 3], step * variables.tau, H.x, H.y, H.z, variables.L, target);
        if (variables.processId == MAIN_PROCESS_ID) {
            cout << "Layer " << step << " max error: " << error << endl;
        }
    }

    // for report.pdf (нарисовать графики)
    // saveValues(u[steps % 3], t, parallelepipeds, "numerical.json", variables);

    // fillDifferenceValues(u[steps % 3], t, variables);
    // saveValues(u[steps % 3], t, parallelepipeds, "difference.json", variables);

    // fillAnalyticalValues(u[0], t, variables);
    // saveValues(u[0], t, parallelepipeds, "analytical.json", variables);
    //

    return evaluateError(u[steps % 3], t, H.x, H.y, H.z, variables.L, target);
}

// ЛАЗАРЕВ В.А. / 628 группа / 2 вариант
int main(int argc, char *argv[]) {
    SolverVariables variables;

    int processId, countOfProcesses;
    // Инициализация MPI, создание группы процессов, создание области связи MPI_COMM_WORLD
    MPI_Init(NULL, NULL);
    // Определяем номер процесса (сохранится в переменную processId)
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    // Определение числа процессов в области связи MPI_COMM_WORLD (сохранится в переменную countOfProcesses)
    MPI_Comm_size(MPI_COMM_WORLD, &countOfProcesses);

    initVariables(argc, argv, variables, processId, countOfProcesses);

    double start = MPI_Wtime();
    double error = 0;
    int loops = 1;

    for (size_t i = 0; i < loops; i++) {
        error += makeSolution(variables);
    }

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
        ofstream fout("results.txt");
        fout << "### Lx = Ly = Lz = " << variables.L.x << ", N = " << variables.N << ", K = " << variables.K << endl
             << endl;
        fout << "| Число MPI процессов (P) | Время решения (с) | Ускорение | Погрешность |" << endl;
        fout << "|                     :-: |               :-: |       :-: |         :-: |" << endl;

        fout << "| " << setw(23) << variables.countOfProcesses;
        fout << " | " << setw(17) << (maxTime / loops);
        fout << " | "
             << "         ";
        fout << " | " << setw(11) << (error / loops);
        fout << " |" << endl;

        fout << "Final error: " << error / loops << endl;
        fout << "Minimal time (s): " << minTime / loops << endl;
        fout << "Maximum time (s): " << maxTime / loops << endl;
        fout << "Average time (s): " << avgTime / loops << endl
             << endl;
        fout.close();

        cout << "Maximum time (s): " << maxTime / loops << endl;
    }

    MPI_Finalize();

    return 0;
}