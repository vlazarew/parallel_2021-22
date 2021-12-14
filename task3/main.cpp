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

// Оси разбиения
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
               vector<ProcessParallelepiped> &subParallelepipeds) {

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
vector<ProcessParallelepiped> splitParallelepiped(int N, int countOfProcesses) {
    vector<ProcessParallelepiped> subParallelepipeds;

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
bool getNeighbours(ProcessParallelepiped first, ProcessParallelepiped second, ProcessParallelepiped &result) {
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
void fillNeighbours(vector<ProcessParallelepiped> &parallelepipeds, SolverVariables &variables) {
    variables.send.clear();
    variables.recv.clear();
    variables.processIds.clear();
    ProcessParallelepiped targetParallelepiped = variables.processParallelepiped;

    for (int i = 0; i < variables.countOfProcesses; i++) {
        // Самого себя не обрабатываем
        if (i == variables.processId) {
            continue;
        }

        ProcessParallelepiped send{};
        ProcessParallelepiped recv{};

        ProcessParallelepiped &processParallelepiped = parallelepipeds[i];
        // Получаем соседей, которым будем отправлять информацию
        if (!getNeighbours(targetParallelepiped, processParallelepiped, send)) {
            continue;
        }

        // Получаем соседей, от которых будем получать информацию
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

// Получить индекс по x, y, z для конкрентного параллелепипеда
int getIndex(int x, int y, int z, ProcessParallelepiped target) {
    return (x - target.xMin) * target.dy * target.dz + (y - target.yMin) * target.dz + (z - target.zMin);
}

// Получить локальный индекс
int getLocalIndex(int x, int y, int z, const SolverVariables &variables) {
    return getIndex(x, y, z, variables.processParallelepiped);
}

double findValue(vector<double> u, int x, int y, int z, vector<vector<double>> recv, SolverVariables variables) {
    for (int index = 0; index < variables.processIds.size(); index++) {
        ProcessParallelepiped parallelepiped = variables.recv[index];

        if (x < parallelepiped.xMin || x > parallelepiped.xMax ||
            y < parallelepiped.yMin || y > parallelepiped.yMax ||
            z < parallelepiped.zMin || z > parallelepiped.zMax) {
            continue;
        }

        return recv[index][getIndex(x, y, z, parallelepiped)];
    }

    return u[getLocalIndex(x, y, z, variables)];
}

// Оператор Лапласа
double calculateLaplaceOperator(vector<double> u, int x, int y, int z, const vector<vector<double>> &recv,
                                const SolverVariables &variables) {
    GridSteps H = variables.H;

    double dx = (findValue(u, x - 1, y, z, recv, variables) + findValue(u, x + 1, y, z, recv, variables) -
                 2 * u[getLocalIndex(x, y, z, variables)]) / (H.x * H.x);
    double dy = (findValue(u, x, y - 1, z, recv, variables) + findValue(u, x, y + 1, z, recv, variables) -
                 2 * u[getLocalIndex(x, y, z, variables)]) / (H.y * H.y);
    double dz = (findValue(u, x, y, z - 1, recv, variables) + findValue(u, x, y, z + 1, recv, variables) -
                 2 * u[getLocalIndex(x, y, z, variables)]) / (H.z * H.z);

    return dx + dy + dz;
}

// Получить значение граничного условия
double getBoundaryValue(int x, int y, int z, double t, const SolverVariables &variables) {
    int N = variables.N;
    double hx = variables.H.x;
    double hy = variables.H.y;
    double hz = variables.H.z;

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
        return getAnalyticValue(x * hx, y * hy, z * hz, t, variables.L);
    }

    return 0;
}

// Заполнение граничными значениями
void fillBoundaryValues(vector<double> &u, double tau, const SolverVariables &variables) {
    int N = variables.N;

    ProcessParallelepiped processParallelepiped = variables.processParallelepiped;

    // Граница при 0
    if (processParallelepiped.xMin == 0) {
        // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
        for (int y = processParallelepiped.yMin; y <= processParallelepiped.yMax; y++) {
            for (int z = processParallelepiped.zMin; z <= processParallelepiped.zMax; z++) {
                // u[индекс по x, y, z для текущего параллелепипеда]
                u[getLocalIndex(processParallelepiped.xMin, y, z, variables)] =
                        getBoundaryValue(processParallelepiped.xMin, y, z, tau, variables);
            }
        }
    }

    // Граница при N
    if (processParallelepiped.xMax == N) {
        // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
        for (int y = processParallelepiped.yMin; y <= processParallelepiped.yMax; y++) {
            for (int z = processParallelepiped.zMin; z <= processParallelepiped.zMax; z++) {
                u[getLocalIndex(processParallelepiped.xMax, y, z, variables)] =
                        getBoundaryValue(processParallelepiped.xMax, y, z, tau, variables);
            }
        }
    }

    // Граница при 0
    if (processParallelepiped.yMin == 0) {
        // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
        for (int x = processParallelepiped.xMin; x <= processParallelepiped.xMax; x++) {
            for (int z = processParallelepiped.zMin; z <= processParallelepiped.zMax; z++) {
                u[getLocalIndex(x, processParallelepiped.yMin, z, variables)] =
                        getBoundaryValue(x, processParallelepiped.yMin, z, tau, variables);
            }
        }
    }

    // Граница при N
    if (processParallelepiped.yMax == N) {
        // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
        for (int x = processParallelepiped.xMin; x <= processParallelepiped.xMax; x++) {
            for (int z = processParallelepiped.zMin; z <= processParallelepiped.zMax; z++) {
                u[getLocalIndex(x, processParallelepiped.yMax, z, variables)] =
                        getBoundaryValue(x, processParallelepiped.yMax, z, tau, variables);
            }
        }
    }

    // Граница при 0
    if (processParallelepiped.zMin == 0) {
        // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
        for (int x = processParallelepiped.xMin; x <= processParallelepiped.xMax; x++) {
            for (int y = processParallelepiped.yMin; y <= processParallelepiped.yMax; y++) {
                u[getLocalIndex(x, y, processParallelepiped.zMin, variables)] =
                        getBoundaryValue(x, y, processParallelepiped.zMin, tau, variables);
            }
        }
    }

    // Граница при N
    if (processParallelepiped.zMax == N) {
        // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
        for (int x = processParallelepiped.xMin; x <= processParallelepiped.xMax; x++)
            for (int y = processParallelepiped.yMin; y <= processParallelepiped.yMax; y++)
                u[getLocalIndex(x, y, processParallelepiped.zMax, variables)] =
                        getBoundaryValue(x, y, processParallelepiped.zMax, tau, variables);
    }
}

#pragma endregion

#pragma region data send recv

// Собрать параллелепипед в обособленный массив (вектор)
vector<double>
packParallelepiped(vector<double> u, ProcessParallelepiped parallelepiped, const SolverVariables &variables) {
    vector<double> packed(parallelepiped.size);

    // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
    for (int i = parallelepiped.xMin; i <= parallelepiped.xMax; i++) {
        for (int j = parallelepiped.yMin; j <= parallelepiped.yMax; j++) {
            for (int k = parallelepiped.zMin; k <= parallelepiped.zMax; k++) {
                packed[getIndex(i, j, k, parallelepiped)] = u[getLocalIndex(i, j, k, variables)];
            }
        }
    }

    return packed;
}

// Отправка и получение соседних значений
vector<vector<double>> sendRecvValues(const vector<double> &u, SolverVariables variables) {
    unsigned long countOfNeighbours = variables.processIds.size();
    vector<vector<double>> recv(countOfNeighbours);

    for (int i = 0; i < countOfNeighbours; i++) {
        vector<double> packed = packParallelepiped(u, variables.send[i], variables);
        recv[i] = vector<double>(variables.recv[i].size);

        vector<MPI_Request> requests(2);
        vector<MPI_Status> statuses(2);

        // Отправляем буффер с начальным адресом packed.data() и кол-вом элементов variables.send[i].size типа double
        // в процесс с id variables.processIds[i] с пометкой 0 в коммутаторе MPI_COMM_WORLD. Выходное значение записывается в requests[0]
        MPI_Isend(packed.data(), variables.send[i].size, MPI_DOUBLE, variables.processIds[i], 0, MPI_COMM_WORLD,
                  &requests[0]);
        // Получаем буффер с начальным адресом recv[i].data() и кол-вом элементов variables.recv[i].size типа double
        // из процесса с id variables.processIds[i] с пометкой 0 в коммутаторе MPI_COMM_WORLD. Выходное значение записывается в requests[1]
        MPI_Irecv(recv[i].data(), variables.recv[i].size, MPI_DOUBLE, variables.processIds[i], 0,
                  MPI_COMM_WORLD, &requests[1]);
        // 2 = размер списка requests. Полученные статусы всех Isend/Irecv записываются в statuses
        // Ожидает выполнения всех MPI_Requests из списка requests
        MPI_Waitall(2, requests.data(), statuses.data());
    }

    return recv;
}

// Отправка и получение общих значений
vector<double>
sendRecvTotal(vector<double> u, vector<ProcessParallelepiped> parallelepipeds, const SolverVariables &variables) {
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
        ProcessParallelepiped &parallelepiped = parallelepipeds[index];
        vector<double> uI(parallelepiped.size);

        if (index == variables.processId) {
            uI = u;
        } else {
            vector<MPI_Request> requests(1);
            vector<MPI_Status> statuses(1);

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

// Заполнение начальных условий
void fillVectorByInitialValues(vector<vector<double>> &u, const SolverVariables &variables) {
    // Заполняем граничные условия
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

    // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
    for (int x = xMin; x <= xMax; x++) {
        for (int y = yMin; y <= yMax; y++) {
            for (int z = zMin; z <= zMax; z++) {
                u[0][getLocalIndex(x, y, z, variables)] = getPhi(x * H.x, y * H.y, z * H.z, variables.L);
            }
        }
    }

    vector<vector<double>> recv = sendRecvValues(u[0], variables);

    // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
    for (int x = xMin; x <= xMax; x++) {
        for (int y = yMin; y <= yMax; y++) {
            for (int z = zMin; z <= zMax; z++) {
                u[1][getLocalIndex(x, y, z, variables)] =
                        u[0][getLocalIndex(x, y, z, variables)] +
                        variables.tau * variables.tau / 2
                        * calculateLaplaceOperator(u[0], x, y, z, recv, variables);
            }
        }
    }
}

// Заполнение следующего слоя
void fillNextLayer(const vector<double> &u0, const vector<double> &u1, vector<double> u, double t,
                   const SolverVariables &variables) {
    int xMin = max(variables.processParallelepiped.xMin, 1);
    int xMax = max(variables.processParallelepiped.xMax, variables.N - 1);

    int yMin = max(variables.processParallelepiped.yMin, 1);
    int yMax = max(variables.processParallelepiped.yMax, variables.N - 1);

    int zMin = max(variables.processParallelepiped.zMin, 1);
    int zMax = max(variables.processParallelepiped.zMax, variables.N - 1);

    vector<vector<double>> recv = sendRecvValues(u1, variables);

    // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
    for (int x = xMin; x <= xMax; x++) {
        for (int y = yMin; y <= yMax; y++) {
            for (int z = zMin; z <= zMax; z++) {
                u[getLocalIndex(x, y, z, variables)] =
                        2 * u1[getLocalIndex(x, y, z, variables)] -
                        u0[getLocalIndex(x, y, z, variables)] +
                        variables.tau * variables.tau * calculateLaplaceOperator(u1, x, y, z, recv, variables);
            }
        }
    }

    fillBoundaryValues(u, t, variables);
}

// Заполнить аналитическими значениями
void fillAnalyticalValues(vector<double> u, double t, const SolverVariables &variables) {
    ProcessParallelepiped parallelepiped = variables.processParallelepiped;
    GridSteps H = variables.H;

    // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
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
void fillDifferenceValues(vector<double> u, double t, const SolverVariables &variables) {
    ProcessParallelepiped parallelepiped = variables.processParallelepiped;
    GridSteps H = variables.H;

    // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel
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

// Оценка погрешности на слое
double evaluateError(vector<double> u, double t, const SolverVariables &variables) {
    double localError = 0, error = 0;
    int N = variables.N;
    GridSteps H = variables.H;
    ProcessParallelepiped parallelepiped = variables.processParallelepiped;

    // https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
    // Выполняется max в переменной localError (как альтернатива можно было б использовать critical секцию)
    // Директива указывает на то, что данный цикл следует разделить по итерациям между потоками.
#pragma omp parallel reduction(max: localError)
    for (int x = parallelepiped.xMin; x <= parallelepiped.xMax; x++) {
        for (int y = parallelepiped.yMin; y <= parallelepiped.yMax; y++) {
            for (int z = parallelepiped.zMin; z <= parallelepiped.zMax; z++) {
                localError = max(localError, fabs(u[getLocalIndex(x, y, z, variables)] -
                                                  getAnalyticValue(x * H.x, y * H.y, z * H.z, t, variables.L)));
            }
        }
    }

    // Отбираем максимальный localError в переменную error по всем процессам
    MPI_Reduce(&localError, &error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return error;
}

// Cохранение слоя в формате json. Для построения графика на спец ресурсе
void
saveValues(vector<double> u, double t, const vector<ProcessParallelepiped> &parallelepipeds, const char *filename,
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
    int layerSize = variables.layerSize;
    int steps = variables.steps;
    int N = variables.N;
    double tau = variables.tau;

    // Разделяем параллелепипед по процессам на sub-параллелепипеды (Step 2)
    vector<ProcessParallelepiped> parallelepipeds = splitParallelepiped(N, variables.countOfProcesses);
    // Запоминаем рабочий параллелепипед
    variables.processParallelepiped = parallelepipeds[variables.processId];

    // Заполняем соседей
    fillNeighbours(parallelepipeds, variables);

    vector<vector<double>> u{vector<double>(layerSize), vector<double>(layerSize), vector<double>(layerSize)};

    // Заполняем начальнные условия (step 4)
    fillVectorByInitialValues(u, variables);

    double error0 = evaluateError(u[0], 0, variables);
    double error1 = evaluateError(u[1], variables.tau, variables);

    if (variables.processId == MAIN_PROCESS_ID) {
        cout << "Layer 0 max error: " << error0 << endl;
        cout << "Layer 1 max error: " << error1 << endl;
    }

    double t = steps * tau;
    for (int step = 2; step <= steps; step++) {
        // Заполняем следующий слой u (step 5 + 6)
        fillNextLayer(u[(step + 1) % 3], u[(step + 2) % 3], u[step % 3], step * variables.tau, variables);

        // Вычисляем максимальную ошибку (step 8)
        double error = evaluateError(u[steps % 3], t, variables);
        if (variables.processId == MAIN_PROCESS_ID) {
            cout << "Layer " << step << " max error: " << error << endl;
        }
    }

    // for report.pdf (нарисовать графики)
    saveValues(u[steps % 3], t, parallelepipeds, "numerical.json", variables);

    fillDifferenceValues(u[steps % 3], t, variables);
    saveValues(u[steps % 3], t, parallelepipeds, "difference.json", variables);

    fillAnalyticalValues(u[0], t, variables);
    saveValues(u[0], t, parallelepipeds, "analytical.json", variables);
    //

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