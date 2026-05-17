
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def solve_2x2(a1, b1, c1, a2, b2, c2):
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return (x, y)


def is_feasible(x1, x2, tol=1e-9):
    if x1 < -tol or x2 < -tol:
        return False
    if -2*x1 + x2 > 1 + tol:
        return False
    if x1 + x2 > 3 + tol:
        return False
    if x1 > 2 + tol:
        return False
    return True


def find_vertices():
    lines = [
        (-2, 1, 1),
        ( 1, 1, 3),
        ( 1, 0, 2),
        ( 1, 0, 0),
        ( 0, 1, 0),
    ]

    vertices = []
    n = len(lines)

    for i in range(n):
        for j in range(i + 1, n):
            pt = solve_2x2(*lines[i], *lines[j])
            if pt is None:
                continue
            x1, x2 = round(pt[0], 8), round(pt[1], 8)
            if is_feasible(x1, x2) and (x1, x2) not in vertices:
                vertices.append((x1, x2))

    return vertices


def objective(x1, x2):
    return 3 * x1 + 2 * x2


def evaluate_vertices(vertices):
    results = [(x1, x2, objective(x1, x2)) for x1, x2 in vertices]
    results.sort(key=lambda r: r[2])
    return results


def print_results(results):
    min_z = results[0][2]
    max_z = results[-1][2]

    print("=" * 50)
    print("  ОЦЕНКА НА ЦЕЛЕВАТА ФУНКЦИЯ z = 3*x1 + 2*x2")
    print("=" * 50)
    print(f"  {'No':>3}  {'x1':>8}  {'x2':>8}  {'z':>8}")
    print("-" * 50)

    for i, (x1, x2, z) in enumerate(results, 1):
        if abs(z - min_z) < 1e-9:
            marker = "  <- MIN"
        elif abs(z - max_z) < 1e-9:
            marker = "  <- MAX"
        else:
            marker = ""
        print(f"  {i:>3}.  {x1:>8.4f}  {x2:>8.4f}  {z:>8.4f}{marker}")

    print("=" * 50)
    print(f"\n  z_min = {min_z:.4f}  при  x1 = {results[0][0]:.4f},  x2 = {results[0][1]:.4f}")
    print(f"  z_max = {max_z:.4f}  при  x1 = {results[-1][0]:.4f},  x2 = {results[-1][1]:.4f}")
    print()
    print("  Интерпретация:")
    print(f"  Минимумът z={min_z:.4f} се достига в (0, 0) — нулево използване на ресурсите.")
    print(f"  Максимумът z={max_z:.4f} се достига при x1=2, x2=1.")
    print("  x1 е максимизиран (коефициент 3 > 2), а останалият")
    print("  капацитет (x1+x2<=3) се запълва с x2=1.")


def plot_solution(vertices, results):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-0.3, 3.2)
    ax.set_ylim(-0.3, 3.5)
    ax.set_xlabel("x1", fontsize=12)
    ax.set_ylabel("x2", fontsize=12)
    ax.set_title("z = 3x1 + 2x2  —  Допустима област и оптимум", fontsize=12)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, linestyle='--', alpha=0.4)

    if len(vertices) >= 3:
        cx = sum(v[0] for v in vertices) / len(vertices)
        cy = sum(v[1] for v in vertices) / len(vertices)
        sorted_v = sorted(vertices, key=lambda v: np.arctan2(v[1] - cy, v[0] - cx))
        px = [v[0] for v in sorted_v] + [sorted_v[0][0]]
        py = [v[1] for v in sorted_v] + [sorted_v[0][1]]
        ax.fill(px, py, alpha=0.2, color='steelblue', label='Допустима област')
        ax.plot(px, py, color='steelblue', linewidth=1.5)

    x_vals = np.linspace(-0.3, 3.2, 300)
    ax.plot(x_vals, 1 + 2*x_vals, 'r--', linewidth=1.2, label='-2x1 + x2 = 1')
    ax.plot(x_vals, 3 - x_vals,   'g--', linewidth=1.2, label='x1 + x2 = 3')
    ax.axvline(x=2, color='purple', linestyle='--', linewidth=1.2, label='x1 = 2')

    min_z = results[0][2]
    max_z = results[-1][2]

    for x1, x2 in vertices:
        z = objective(x1, x2)
        ax.plot(x1, x2, 'ko', markersize=6, zorder=5)
        ax.annotate(f"({x1:.2f}, {x2:.2f})\nz={z:.2f}",
                    xy=(x1, x2), xytext=(x1 + 0.07, x2 + 0.07), fontsize=8)

    for x1, x2, z in results:
        if abs(z - min_z) < 1e-9:
            ax.plot(x1, x2, 'b*', markersize=16, zorder=6, label=f'MIN z={min_z:.2f}')
        if abs(z - max_z) < 1e-9:
            ax.plot(x1, x2, 'r*', markersize=16, zorder=6, label=f'MAX z={max_z:.2f}')

    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig("zadacha_1_grafik.png", dpi=150)
    print("  Графиката е запазена като 'zadacha_1_grafik.png'")


def main():
    print("\n" + "=" * 50)
    print("  ЗАДАЧА 1 — ЛИНЕЙНА ОПТИМИЗАЦИЯ")
    print("=" * 50)

    vertices = find_vertices()

    if not vertices:
        print("  Няма допустими върхове — задачата е неразрешима.")
        return

    results = evaluate_vertices(vertices)
    print_results(results)
    plot_solution(vertices, results)


if __name__ == "__main__":
    main()