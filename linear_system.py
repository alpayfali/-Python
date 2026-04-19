import numpy as np
import csv

# ── Константа за нулев праг ───────────────────────────────────────────────────

EPSILON = 1e-10


# ── Четене на входни данни от CSV ─────────────────────────────────────────────

def read_csv(filename):
    rows = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([float(x) for x in row])

    A = []
    B = []
    for row in rows:
        A.append(row[:-1])   # всички числа без последното
        B.append(row[-1])    # само последното число

    return np.array(A, dtype=float), np.array(B, dtype=float)


# ── Зануляване на почти-нулеви стойности ─────────────────────────────────────

def clean(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs(matrix[i, j]) < EPSILON:
                matrix[i, j] = 0.0
    return matrix


# ── Намиране на pivot елемент в колона ───────────────────────────────────────

def find_pivot(matrix, col, start_row):
    best_row = -1
    best_val = EPSILON

    for row in range(start_row, matrix.shape[0]):
        val = abs(matrix[row, col])
        if val > best_val:
            best_val = val
            best_row = row

    return best_row


# ── Елиминиране с помощта на pivot елемент ────────────────────────────────────

def eliminate(matrix, pivot_row, pivot_col):
    pivot_val = matrix[pivot_row, pivot_col]

    for row in range(matrix.shape[0]):
        if row == pivot_row:
            continue
        if abs(matrix[row, pivot_col]) < EPSILON:
            continue
        factor = matrix[row, pivot_col] / pivot_val
        matrix[row] = matrix[row] - factor * matrix[pivot_row]

    return clean(matrix)


# ── Метод на Гаус ─────────────────────────────────────────────────────────────

def gauss(A, B):
    num_rows = A.shape[0]
    num_cols = A.shape[1]

    # Строим разширената матрица [A|B] ръчно ред по ред
    aug = np.zeros((num_rows, num_cols + 1))
    for i in range(num_rows):
        for j in range(num_cols):
            aug[i, j] = A[i, j]
        aug[i, num_cols] = B[i]

    aug = clean(aug)

    pivot_cols = []
    current_row = 0

    for col in range(num_cols):
        pivot_row = find_pivot(aug, col, current_row)

        if pivot_row == -1:
            continue  # няма pivot → свободна променлива

        # Разменяме редовете
        aug[[current_row, pivot_row]] = aug[[pivot_row, current_row]]

        # Нормализираме pivot реда така че pivot = 1
        aug[current_row] = aug[current_row] / aug[current_row, col]
        aug = clean(aug)

        # Елиминираме всички останали редове
        aug = eliminate(aug, current_row, col)

        pivot_cols.append(col)
        current_row = current_row + 1

    return aug, pivot_cols


# ── Проверка дали системата има решение ──────────────────────────────────────

def is_consistent(aug, num_vars):
    for i in range(aug.shape[0]):
        row = aug[i]
        all_zero = True
        for j in range(num_vars):
            if abs(row[j]) >= EPSILON:
                all_zero = False
        if all_zero and abs(row[num_vars]) >= EPSILON:
            return False
    return True


# ── Извеждане на резултата ────────────────────────────────────────────────────

def print_solution(aug, pivot_cols, num_vars):
    free_cols = []
    for c in range(num_vars):
        if c not in pivot_cols:
            free_cols.append(c)

    print("\n── Редуцирана разширена матрица [A|B] ──")
    print(np.round(aug, 6))

    # Частно решение: свободните променливи = 0
    particular = np.zeros(num_vars)
    for i in range(len(pivot_cols)):
        pc = pivot_cols[i]
        if i < aug.shape[0]:
            particular[pc] = aug[i, num_vars]

    print(f"\n  Свободни променливи: {['x' + str(c + 1) for c in free_cols]}")
    print("\n  Частно решение (свободни = 0):")
    for i in range(num_vars):
        print(f"    x{i + 1} = {particular[i]:.6f}")

    # Базисни вектори — по един за всяка свободна променлива
    print("\n  Базис на общото решение:")
    for fc in free_cols:
        vec = np.zeros(num_vars)
        vec[fc] = 1.0
        for i in range(len(pivot_cols)):
            pc = pivot_cols[i]
            if i < aug.shape[0]:
                vec[pc] = -aug[i, fc]
        vec = clean(vec)
        print(f"    t{fc + 1} * {np.round(vec, 6)}")

    print("\n  Общо решение: x = частно + линейна комбинация на базисните вектори")


# ── Основна програма ──────────────────────────────────────────────────────────

filename = input("Въведи името на CSV файла (напр. system.csv): ").strip()

A, B = read_csv(filename)

print(f"\nМатрица A:\n{A}")
print(f"\nВектор B:\n{B}")

num_rows = A.shape[0]
num_vars = A.shape[1]

aug, pivot_cols = gauss(A, B)

if not is_consistent(aug, num_vars):
    print("\n✘ Системата НЯМА решение (несъвместима).")

elif len(pivot_cols) == num_vars:
    x = np.linalg.solve(A, B)
    print("\n✔ Системата има ЕДИНСТВЕНО решение:")
    for i in range(num_vars):
        print(f"  x{i + 1} = {x[i]:.6f}")

else:
    print("\n✔ Системата има БЕЗКРАЙНО МНОГО решения.")
    print_solution(aug, pivot_cols, num_vars)
