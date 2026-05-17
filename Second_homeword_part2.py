
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#Reading data

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(v) for v in line.split(',')]
            data.append(values)
    return np.array(data)


# Statistics

def compute_stats(data):
    n, m = data.shape
    mu = np.zeros(m)
    for j in range(m):
        mu[j] = sum(data[i, j] for i in range(n)) / n

    cov = np.zeros((m, m))
    for j in range(m):
        for k in range(m):
            s = sum((data[i, j] - mu[j]) * (data[i, k] - mu[k]) for i in range(n))
            cov[j, k] = s / (n - 1)

    return mu, cov


# -- 3. Оптимален портфейл

def project_simplex(w):
    """Проектира вектора w върху симплекса {w >= 0, sum(w)=1}."""
    m = len(w)
    u = sorted(w, reverse=True)
    rho = 0
    cumsum = 0.0
    for j in range(m):
        cumsum += u[j]
        if u[j] - (cumsum - 1) / (j + 1) > 0:
            rho = j
    lam = (sum(u[:rho + 1]) - 1) / (rho + 1)
    return np.array([max(wi - lam, 0.0) for wi in w])


def portfolio_variance(w, cov):
    return float(w @ cov @ w)


def portfolio_return(w, mu):
    return float(w @ mu)


def minimize_variance(cov, mu, min_return=None, lr=0.01, max_iter=50000, tol=1e-10):
    """
    Градиентен спуск с проекция върху симплекса.
    Ако min_return е зададено, прилага penalty за нарушения на ограничението.
    """
    m = len(mu)
    w = np.array([1.0 / m] * m)

    penalty = 500.0

    for iteration in range(max_iter):
        grad = 2.0 * cov @ w

        if min_return is not None:
            ret = portfolio_return(w, mu)
            if ret < min_return:
                grad -= penalty * mu

        w_new = project_simplex(w - lr * grad)

        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new

    return w


# Резултати

def print_portfolio(w, mu, cov, label=""):
    var = portfolio_variance(w, cov)
    ret = portfolio_return(w, mu)
    std = var ** 0.5

    print(f"\n{'=' * 52}")
    print(f"  {label}")
    print(f"{'=' * 52}")
    print(f"  {'Инструмент':>12}  {'Тегло (%)':>10}")
    print(f"  {'-' * 30}")
    for i, wi in enumerate(w):
        print(f"  {'Инстр. ' + str(i+1):>12}  {wi * 100:>10.4f}%")
    print(f"  {'-' * 30}")
    print(f"  {'Сума:':>12}  {sum(w) * 100:>10.4f}%")
    print(f"\n  Дисперсия на портфейла  : {var:.6f}")
    print(f"  Стандартно отклонение   : {std:.6f}")
    print(f"  Очаквана възвращаемост  : {ret:.6f}  ({ret * 100:.2f}%)")


# ---------- 5. Графика ----------

def plot_portfolios(w_a, w_b, mu, cov, out_path):
    labels = [f"Инстр.{i+1}" for i in range(len(mu))]
    x = np.arange(len(mu))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, w, title in zip(
        axes,
        [w_a, w_b],
        ["А: Мин. риск (без ограничение)", "Б: Мин. риск (мин. 20% доход)"]
    ):
        bars = ax.bar(x, w * 100, width, color='steelblue', edgecolor='white')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Инструмент", fontsize=10)
        ax.set_ylabel("Тегло (%)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, max(max(w_a), max(w_b)) * 110)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}%", ha='center', va='bottom', fontsize=8)

    fig.suptitle("Оптимални портфейли — разпределение на теглата", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\n  Графиката е запазена като '{out_path}'")


# ---------- Главна функция ----------

def main():
    filepath =  Path(__file__).parent / "observations(in).csv"

    print("=" * 52)
    print("  ЗАДАЧА 2 — ОПТИМАЛЕН ПОРТФЕЙЛ")
    print("=" * 52)

    try:
        data = load_data(filepath)
    except FileNotFoundError:
        print(f"  Файлът '{filepath}' не е намерен в текущата директория.")
        print("  Постави файла до скрипта и стартирай отново.")
        return
    n, m = data.shape
    print(f"\n  Заредени данни: {n} наблюдения, {m} инструмента")

    mu, cov = compute_stats(data)

    print("\n  Средни възвращаемости (mu):")
    for i, v in enumerate(mu):
        print(f"    Инстр. {i+1}: {v:>10.6f}")

    print("\n  Ковариационна матрица:")
    for row in cov:
        print("   ", "  ".join(f"{v:8.4f}" for v in row))

    # Вариант А — без ограничение за доход
    w_a = minimize_variance(cov, mu)
    print_portfolio(w_a, mu, cov, label="ВАРИАНТ А — Минимален риск (без ограничение)")

    # Вариант Б — мин. 20% очакван доход
    w_b = minimize_variance(cov, mu, min_return=0.20)
    print_portfolio(w_b, mu, cov, label="ВАРИАНТ Б — Минимален риск (мин. 20% доход)")

    ret_b = portfolio_return(w_b, mu)
    if ret_b < 0.20 - 1e-4:
        print("\n  ВНИМАНИЕ: Ограничението за 20% не може да бъде изпълнено")
        print(f"  с тези данни. Най-добрият постигнат доход: {ret_b*100:.4f}%")

    out_path = "zadacha_2_grafik.png"
    plot_portfolios(w_a, w_b, mu, cov, out_path)


if __name__ == "__main__":
    main()