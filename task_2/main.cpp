#include <iostream>
#include <mpi/mpi.h>
#include <string>
#include <cstring>
#include <cmath>

using namespace std;

// Количество точек
const int COUNT_OF_POINTS = 5000;
// Результат аналитического решения интеграла
const double ANALYTIC_VALUE = (16 / (double) 135);

const int MAIN_PROCESS_ID = 0;

// Параллелепипед
const double X_MIN = -1;
const double X_MAX = 1;
const double Y_MIN = -1;
const double Y_MAX = 1;
const double Z_MIN = -2;
const double Z_MAX = 2;

// Объем параллелепипеда П
const double VOLUME_OF_P = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * (Z_MAX - Z_MIN);

void initVariables(int argc, char *argv[], double &eps) {
    // Первый параметр - ссылка на сборку
    for (int i = 1; i < argc; ++i) {
        string currentArgument(argv[i]);
        const char *epsArgKey = "-eps=";
        int espArgName = currentArgument.find(epsArgKey);
        if (espArgName != string::npos) {
            string epsString = currentArgument.substr(espArgName + strlen(epsArgKey));
            try {
                eps = atof(epsString.c_str());
            } catch (...) {
                throw runtime_error("Invalid input epsilon");
            }
            if (eps < 0.00000001 || eps > 1) {
                throw runtime_error("Invalid input epsilon");
            }
        }
        // Остальные параметры будут игнорироваться (ну или позже добавлю какие-нибудь свои кастомные)
    }
}

// Генерация случайных значений (точки) по определенной оси в указанном интервале
double GenerateAxisValue(double minValue, double maxValue) {
    double randomCoefficient = (double) rand() / RAND_MAX;
    return (randomCoefficient * (maxValue - minValue)) + minValue;
}

// Подинтегральное выражение
double IntegrandFunction(double x, double y, double z) {
    if ((fabs(x) + fabs(y)) <= 1 && z >= -2 && z <= 2) {
        return x * x * y * y * z * z;
    }
    return 0;
}

// Посчитать результат на процессе
double CalculateValue() {
    double result = 0;

    for (int i = 0; i < COUNT_OF_POINTS; ++i) {
        double x = GenerateAxisValue(X_MIN, X_MAX);
        double y = GenerateAxisValue(Y_MIN, Y_MAX);
        double z = GenerateAxisValue(Z_MIN, Z_MAX);

        result += IntegrandFunction(x, y, z);
    }

    return result / COUNT_OF_POINTS;
}

double makeSolution(int processId, int countOfProcesses, double &eps, int &countOfIterations, double &calculatedSum) {
    double integratedValue = 0;

    bool haveResult = false;

    while (!haveResult) {
        double resultForProcess = CalculateValue();
        double sumOfResults = 0;

        MPI_Reduce(&resultForProcess, &sumOfResults, 1, MPI_DOUBLE, MPI_SUM, MAIN_PROCESS_ID, MPI_COMM_WORLD);

        if (processId == MAIN_PROCESS_ID) {
            countOfIterations++;
            calculatedSum += sumOfResults / countOfProcesses;
            integratedValue = calculatedSum * VOLUME_OF_P / countOfIterations;

            double diff = fabs(ANALYTIC_VALUE - integratedValue);
            if (true) {
                cout << "Loop - " << countOfIterations << ".  Integrated Value = " << integratedValue <<
                     "; diff = " << diff << endl;
            }

            haveResult = (diff <= eps);
        }

        // Отправляем во все процессы флажок haveResult ("Нужно ли производить еще одну итерацию цикла"). Источник значения: MAIN_PROCESS_ID
        MPI_Bcast(&haveResult, 1, MPI_CXX_BOOL, MAIN_PROCESS_ID, MPI_COMM_WORLD);
    }

    return integratedValue;
}

void printStats(int countOfProcesses, double eps, int countOfIterations, double integratedValue, double minTime,
                double maxTime, double avgTime) {
    cout << "--------------------------------------------------------------" << endl;
    cout << "Author: Lazarev V. A. / 8 Option / Independent generation of points by MPI processes" << endl;
    cout << "Count of processes: " << countOfProcesses << endl;
    cout << "Epsilon: " << eps << endl;
    cout << "Count of points (for each process): " << COUNT_OF_POINTS << endl;
    cout << "x = [" << X_MIN << ", " << X_MAX << "]; y = [" << Y_MIN << ", " << Y_MAX << "]; z = [" << Z_MIN << ", "
         << Z_MAX << "];" << endl;
    cout << "VOLUME_OF_P: " << VOLUME_OF_P << endl;
    cout << "--------------------------------------------------------------" << endl;
    cout << "Count of iterations: " << countOfIterations << endl;
    cout << "Total count of poins: " << COUNT_OF_POINTS * countOfIterations * countOfProcesses << endl;
    cout << "Analytic value: " << ANALYTIC_VALUE << endl;
    cout << "Calculated value : " << integratedValue << endl;
    cout << "Error: " << fabs(ANALYTIC_VALUE - integratedValue) << endl;
    cout << "--------------------------------------------------------------" << endl;
    cout << "Minimal time (s): " << minTime << endl;
    cout << "Maximum time (s): " << maxTime << endl;
    cout << "Average time (s): " << avgTime << endl;
}

// ЛАЗАРЕВ В.А. / 628 группа / 8 вариант
int main(int argc, char *argv[]) {
    // Точность решения
    double eps = 0.0001;
    initVariables(argc, argv, eps);

    int processId, countOfProcesses;
    int countOfIterations = 0;
    double calculatedSum = 0;

    // Инициализация MPI, создание группы процессов, создание области связи MPI_COMM_WORLD
    MPI_Init(&argc, &argv);
    // Определяем номер процесса (сохранится в переменную processId)
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    // Определение числа процессов в области связи MPI_COMM_WORLD (сохранится в переменную countOfProcesses)
    MPI_Comm_size(MPI_COMM_WORLD, &countOfProcesses);

    srand(processId);

    double start = MPI_Wtime();

    double integratedValue = makeSolution(processId, countOfProcesses, eps, countOfIterations, calculatedSum);

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
        printStats(countOfProcesses, eps, countOfIterations, integratedValue, minTime, maxTime, avgTime);
    }

    MPI_Finalize();
    return 0;
}
