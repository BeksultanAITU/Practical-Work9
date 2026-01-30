#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr double EPS_PIVOT = 1e-12;


template <typename T>
static void print_matrix_block(const std::vector<T>& a, int n, int rows, int cols, const std::string& title) {
    std::cout << title << "\n";
    for (int i = 0; i < std::min(n, rows); ++i) {
        for (int j = 0; j < std::min(n, cols); ++j) {
            std::cout << std::setw(8) << a[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}


// TASK 1: Scatterv counts/displs (учёт остатка N % size)

static void make_counts_displs(int N, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);

    int base = N / size;
    int rem  = N % size;

    for (int i = 0; i < size; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + counts[i - 1]);
    }
}


// TASK 2: Генерация диагонально-доминантной матрицы (без нулевых pivot)
// Это упрощает Гаусс без сложного распределённого выбора главного элемента.

static void generate_diagonally_dominant_system(int N, std::vector<double>& A, std::vector<double>& b) {
    A.assign((size_t)N * N, 0.0);
    b.assign(N, 0.0);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < N; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            double v = dist(rng);
            A[(size_t)i * N + j] = v;
            row_sum += std::fabs(v);
        }
        // Диагональ делаем достаточно большой, чтобы pivot почти всегда был != 0
        A[(size_t)i * N + i] = row_sum + 5.0;

        b[i] = dist(rng);
    }
}


// TASK 2: Back substitution на rank 0 (после прямого хода)

static std::vector<double> back_substitution(int N, const std::vector<double>& U, const std::vector<double>& y) {
    std::vector<double> x(N, 0.0);

    for (int i = N - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < N; ++j) {
            s -= U[(size_t)i * N + j] * x[j];
        }
        double piv = U[(size_t)i * N + i];
        x[i] = s / piv;
    }
    return x;
}


// TASK 2: Норма невязки ||Ax-b||_2 для проверки качества решения

static double residual_norm2(int N, const std::vector<double>& A, const std::vector<double>& b,
                             const std::vector<double>& x)
{
    double acc = 0.0;
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) {
            s += A[(size_t)i * N + j] * x[j];
        }
        double r = s - b[i];
        acc += r * r;
    }
    return std::sqrt(acc);
}


// TASK 3: Генерация матрицы смежности (Floyd–Warshall)
// INF означает "нет ребра". Диагональ = 0.

static void generate_graph(int N, std::vector<int>& G) {
    const int INF = 1'000'000'000;
    G.assign((size_t)N * N, INF);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<int> wdist(1, 20);

    for (int i = 0; i < N; ++i) {
        G[(size_t)i * N + i] = 0;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            // вероятность ребра ~ 0.35 (для демо)
            if (prob(rng) < 0.35) {
                G[(size_t)i * N + j] = wdist(rng);
            }
        }
    }
}


// MAIN

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Параметры (можно переопределить через аргументы)
    int N1 = 1'000'000;  // TASK 1
    int N2 = 256;        // TASK 2 (для Гаусса лучше не огромный, иначе долго)
    int N3 = 256;        // TASK 3 (для Флойда лучше не огромный, иначе O(N^3))

    if (argc >= 2) N1 = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) N2 = std::max(2, std::atoi(argv[2]));
    if (argc >= 4) N3 = std::max(2, std::atoi(argv[3]));

    if (rank == 0) {
        std::cout << "Practical_Work9\n";
        std::cout << "MPI processes: " << size << "\n";
        std::cout << "N_task1 = " << N1 << ", N_task2 = " << N2 << ", N_task3 = " << N3 << "\n\n";
    }

    
    // TASK 1: Mean + StdDev (Scatterv + Reduce)
    
    {
        if (rank == 0) {
            std::cout << "TASK 1\n";
            std::cout << "Distributed mean and standard deviation (Scatterv + Reduce)\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        std::vector<double> full;
        if (rank == 0) {
            full.resize(N1);

            // Генерация данных на root
            std::mt19937 rng(12345);
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (int i = 0; i < N1; ++i) full[i] = dist(rng);
        }

        // Считаем, сколько элементов получит каждый процесс
        std::vector<int> counts, displs;
        make_counts_displs(N1, size, counts, displs);

        int local_n = counts[rank];
        std::vector<double> local(local_n);

        // Scatterv: корректно раздаёт массив даже если N не делится на size
        MPI_Scatterv(
            rank == 0 ? full.data() : nullptr,
            counts.data(),
            displs.data(),
            MPI_DOUBLE,
            local.data(),
            local_n,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        // Каждый процесс считает сумму и сумму квадратов своей части
        double local_sum = 0.0;
        double local_sumsq = 0.0;
        for (double x : local) {
            local_sum += x;
            local_sumsq += x * x;
        }

        // Reduce: собираем локальные суммы на root
        double total_sum = 0.0;
        double total_sumsq = 0.0;
        MPI_Reduce(&local_sum,   &total_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sumsq, &total_sumsq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0) {
            double mean = total_sum / (double)N1;
            double var  = total_sumsq / (double)N1 - mean * mean;
            if (var < 0.0) var = 0.0;
            double stddev = std::sqrt(var);

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "N                 : " << N1 << "\n";
            std::cout << "Mean              : " << mean << "\n";
            std::cout << "StdDev            : " << stddev << "\n";
            std::cout << "Execution time    : " << std::setprecision(6) << (t1 - t0) << " seconds\n";
            std::cout << "\n";
        }
    }

    
    // TASK 2: Distributed Gaussian Elimination (Scatter + Bcast + Gather)
  
    
    {
        if (rank == 0) {
            std::cout << "TASK 2\n";
            std::cout << "Distributed Gaussian elimination (Scatter + Bcast + Gather)\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        int rows_per_proc = (N2 + size - 1) / size;
        int padded_rows   = rows_per_proc * size;

        // Root создаёт A и b
        std::vector<double> A_full, b_full;
        std::vector<double> A_orig, b_orig; // для проверки невязки (на root)
        if (rank == 0) {
            std::vector<double> A0, b0;
            generate_diagonally_dominant_system(N2, A0, b0);

            A_orig = A0;
            b_orig = b0;

            // Паддинг до padded_rows
            A_full.assign((size_t)padded_rows * N2, 0.0);
            b_full.assign(padded_rows, 0.0);

            for (int i = 0; i < N2; ++i) {
                std::copy(&A0[(size_t)i * N2], &A0[(size_t)i * N2 + N2], &A_full[(size_t)i * N2]);
                b_full[i] = b0[i];
            }
        }

        // Локальные куски A и b (у каждого процесса одинаковый размер для Scatter)
        std::vector<double> A_local((size_t)rows_per_proc * N2, 0.0);
        std::vector<double> b_local(rows_per_proc, 0.0);

        // Scatter строк матрицы: каждый процесс получает rows_per_proc строк
        MPI_Scatter(
            rank == 0 ? A_full.data() : nullptr,
            rows_per_proc * N2,
            MPI_DOUBLE,
            A_local.data(),
            rows_per_proc * N2,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        // Scatter правой части
        MPI_Scatter(
            rank == 0 ? b_full.data() : nullptr,
            rows_per_proc,
            MPI_DOUBLE,
            b_local.data(),
            rows_per_proc,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        // Буфер для pivot row (передаётся всем через Bcast)
        std::vector<double> pivot_row(N2, 0.0);
        double pivot_b = 0.0;

        // Прямой ход Гаусса
        for (int k = 0; k < N2; ++k) {
            int owner = k / rows_per_proc;              // у кого находится строка k
            int local_k = k % rows_per_proc;

            if (rank == owner) {
                // Owner берёт pivot строку из своего локального массива
                std::copy(
                    &A_local[(size_t)local_k * N2],
                    &A_local[(size_t)local_k * N2 + N2],
                    pivot_row.begin()
                );
                pivot_b = b_local[local_k];

                // Доп. защита от слишком маленького pivot (в демо почти не нужно)
                if (std::fabs(pivot_row[k]) < EPS_PIVOT) {
                    pivot_row[k] = (pivot_row[k] >= 0.0) ? EPS_PIVOT : -EPS_PIVOT;
                }
            }

            // Передаём pivot строку всем процессам
            MPI_Bcast(pivot_row.data(), N2, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            // Каждый процесс обновляет свои строки (global_i > k)
            double piv = pivot_row[k];

            for (int r = 0; r < rows_per_proc; ++r) {
                int global_i = rank * rows_per_proc + r;
                if (global_i <= k) continue;
                if (global_i >= N2) continue; // паддинг-строки не трогаем

                double aik = A_local[(size_t)r * N2 + k];
                double factor = aik / piv;

                // Вычитаем factor * pivot_row из текущей строки
                // Начинаем с k, чтобы быстрее занулять левую часть
                for (int j = k; j < N2; ++j) {
                    A_local[(size_t)r * N2 + j] -= factor * pivot_row[j];
                }
                b_local[r] -= factor * pivot_b;

                // Явно зануляем (аккуратнее по числам)
                A_local[(size_t)r * N2 + k] = 0.0;
            }
        }

        // Собираем U и y на root
        std::vector<double> U_full, y_full;
        if (rank == 0) {
            U_full.resize((size_t)padded_rows * N2, 0.0);
            y_full.resize(padded_rows, 0.0);
        }

        MPI_Gather(
            A_local.data(),
            rows_per_proc * N2,
            MPI_DOUBLE,
            rank == 0 ? U_full.data() : nullptr,
            rows_per_proc * N2,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Gather(
            b_local.data(),
            rows_per_proc,
            MPI_DOUBLE,
            rank == 0 ? y_full.data() : nullptr,
            rows_per_proc,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0) {
            // Берём только первые N2 строк (без паддинга)
            std::vector<double> U((size_t)N2 * N2, 0.0);
            std::vector<double> y(N2, 0.0);

            for (int i = 0; i < N2; ++i) {
                std::copy(
                    &U_full[(size_t)i * N2],
                    &U_full[(size_t)i * N2 + N2],
                    &U[(size_t)i * N2]
                );
                y[i] = y_full[i];
            }

            // Обратный ход на root
            std::vector<double> x = back_substitution(N2, U, y);

            // Проверка решения (невязка)
            double rnorm = residual_norm2(N2, A_orig, b_orig, x);

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "N                 : " << N2 << "\n";
            std::cout << "Residual norm     : " << rnorm << "\n";
            std::cout << "Execution time    : " << std::setprecision(6) << (t1 - t0) << " seconds\n";

            // Печать первых элементов решения (чтобы не засорять консоль)
            std::cout << "Solution x (first 10): ";
            for (int i = 0; i < std::min(10, N2); ++i) std::cout << x[i] << " ";
            std::cout << "\n\n";
        }
    }

    
    // TASK 3: Floyd–Warshall (Scatter + Allgather)
    
    {
        if (rank == 0) {
            std::cout << "TASK 3\n";
            std::cout << "Parallel Floyd-Warshall shortest paths (Scatter + Allgather)\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        const int INF = 1'000'000'000;

        int rows_per_proc = (N3 + size - 1) / size;
        int padded_rows   = rows_per_proc * size;

        std::vector<int> G_full;
        if (rank == 0) {
            std::vector<int> G0;
            generate_graph(N3, G0);

            // Паддинг: лишние строки INF, диагональ можно 0 (но неважно, т.к. i>=N не используем)
            G_full.assign((size_t)padded_rows * N3, INF);
            for (int i = 0; i < padded_rows; ++i) {
                if (i < N3) {
                    std::copy(
                        &G0[(size_t)i * N3],
                        &G0[(size_t)i * N3 + N3],
                        &G_full[(size_t)i * N3]
                    );
                } else {
                    // паддинг-строка: все INF
                    for (int j = 0; j < N3; ++j) G_full[(size_t)i * N3 + j] = INF;
                }
            }
        }

        // Локальные строки
        std::vector<int> local((size_t)rows_per_proc * N3, INF);

        MPI_Scatter(
            rank == 0 ? G_full.data() : nullptr,
            rows_per_proc * N3,
            MPI_INT,
            local.data(),
            rows_per_proc * N3,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        // Буфер "полной матрицы", который будет заполняться через Allgather
        std::vector<int> global((size_t)padded_rows * N3, INF);

        // Итерации Флойда–Уоршелла
        for (int k = 0; k < N3; ++k) {
            // Обмениваемся данными между процессами: каждый отдаёт свой блок строк
            MPI_Allgather(
                local.data(),
                rows_per_proc * N3,
                MPI_INT,
                global.data(),
                rows_per_proc * N3,
                MPI_INT,
                MPI_COMM_WORLD
            );

            // Теперь у каждого процесса есть актуальная матрица (в global),
            // обновляем только свои строки
            for (int r = 0; r < rows_per_proc; ++r) {
                int i = rank * rows_per_proc + r;
                if (i >= N3) continue;

                int dik = global[(size_t)i * N3 + k];
                if (dik >= INF) continue;

                for (int j = 0; j < N3; ++j) {
                    int dkj = global[(size_t)k * N3 + j];
                    if (dkj >= INF) continue;

                    long long via = (long long)dik + (long long)dkj;
                    int cur = local[(size_t)r * N3 + j];
                    if (via < cur) local[(size_t)r * N3 + j] = (int)via;
                }
            }
        }

        // Собираем итоговую матрицу на root
        std::vector<int> result;
        if (rank == 0) result.resize((size_t)padded_rows * N3, INF);

        MPI_Gather(
            local.data(),
            rows_per_proc * N3,
            MPI_INT,
            rank == 0 ? result.data() : nullptr,
            rows_per_proc * N3,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0) {
            std::cout << "N                 : " << N3 << "\n";
            std::cout << "Execution time    : " << std::fixed << std::setprecision(6) << (t1 - t0) << " seconds\n";

            // Печатаем небольшой блок, чтобы консоль не была огромной
            std::vector<int> trimmed((size_t)N3 * N3, INF);
            for (int i = 0; i < N3; ++i) {
                std::copy(
                    &result[(size_t)i * N3],
                    &result[(size_t)i * N3 + N3],
                    &trimmed[(size_t)i * N3]
                );
            }

            print_matrix_block(trimmed, N3, 8, 8, "Distance matrix (top-left 8x8):");
        }
    }

    if (rank == 0) {
        std::cout << "Done.\n";
    }

    MPI_Finalize();
    return 0;
}
