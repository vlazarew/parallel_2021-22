#include <iostream>
#include <mpi/mpi.h>
#include <vector>
#include <string>
#include <string.h>

using namespace std;

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

// ЛАЗАРЕВ В.А. / 628 группа / 8 вариант
int main(int argc, char *argv[]) {
    // Точность решения
    double eps = 0.0001;
    // Количество точек
    int countOfPoints = 5000;

    // Результат аналитического решения интеграла
    double analyticValue = (16 / 135);
    // Параллелепипед
    double xMin = -1;
    double xMax = 1;
    double yMin = -1;
    double yMax = 1;
    double zMin = -2;
    double zMax = 2;

    // Объем параллелепипеда П
    double volumeOfParal = (xMax - xMin) * (yMax - yMin) * (zMax - zMin);

    initVariables(argc, argv, eps);

    int processId, countOfProcesses;

    // Инициализация MPI, создание группы процессов, создание области связи MPI_COMM_WORLD
    MPI_Init(&argc, &argv);
    // Определяем номер процесса (сохранится в переменную processId)
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    // Определение числа процессов в области связи MPI_COMM_WORLD (сохранится в переменную countOfProcesses)
    MPI_Comm_size(MPI_COMM_WORLD, &countOfProcesses);

    srand(processId);

    cout << "Program finished! ProcessId: " << processId << endl;
    return 0;
}
