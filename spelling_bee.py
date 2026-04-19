# ── Четене на речника ─────────────────────────────────────────────────────────

def read_words():
    words = []
    with open("words.txt", "r") as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    return words


# ── Проверка и оценка на една дума ───────────────────────────────────────────

def score_word(word, letters):
    center = letters[0]

    # Трябва да съдържа централната буква
    if center not in word:
        return 0

    # Не може да съдържа букви извън позволените
    for letter in word:
        if letter not in letters:
            return 0

    # Минимум 4 букви
    if len(word) < 4:
        return 0

    # Точки за дължина
    if len(word) == 4:
        points = 1
    else:
        points = len(word)

    # Бонус за панграм (думата съдържа всичките 7 букви)
    is_pangram = True
    for letter in letters:
        if letter not in word:
            is_pangram = False
    if is_pangram:
        points = points + 7

    return points


# ── Намиране на всички решения ────────────────────────────────────────────────

def find_solutions(letters, words):
    solutions = {}
    for word in words:
        points = score_word(word, letters)
        if points > 0:
            solutions[word] = points
    return solutions


# ── Извеждане на резултата ────────────────────────────────────────────────────

def print_results(solutions, letters):
    print(f"\nЦентрална буква : {letters[0].upper()}")
    print(f"Всички букви    : {' '.join(letters.upper())}\n")

    print(f"{'Дума':<20} {'Точки':>6}  ")
    print("-" * 32)

    sorted_words = sorted(solutions, key=lambda w: solutions[w], reverse=True)

    for word in sorted_words:
        points = solutions[word]
        label = "ПАНГРАМ" if points >= len(word) + 7 else ""
        print(f"{word:<20} {points:>6}  {label}")

    print("-" * 32)
    print(f"Общо думи  : {len(solutions)}")
    print(f"Общо точки : {sum(solutions.values())}")


# ── Стартиране ────────────────────────────────────────────────────────────────

S = input("Въведи 7-те букви (първата е централната): ").strip().lower()

if len(S) != 7 or not S.isalpha():
    print("Грешка: моля въведи точно 7 латински букви.")
else:
    words = read_words()
    solutions = find_solutions(S, words)
    print_results(solutions, S)
