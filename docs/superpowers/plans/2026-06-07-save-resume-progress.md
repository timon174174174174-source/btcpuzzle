# Save/Resume Scan Progress — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** При закрытии сохранять прогресс сканирования диапазона на диск, а при старте — досканировать прерванный диапазон с сохранённой позиции и засабмитить proof-ключи в пул, не теряя до-крашевую GPU-работу.

**Architecture:** Периодический чекпойнт (offset + идентификатор диапазона + найденные proof-ключи) пишется атомарно из мониторящего потока `Search()`. При старте, до запроса нового диапазона, `main()` досканирует сохранённый диапазон (точный resume при той же геометрии GPU, иначе полный рескан) и сабмитит. Чистые модули `Checkpoint` (сериализация) и `RangeMath` (арифметика партиций) покрыты юнит-тестами без GPU.

**Tech Stack:** C++17, CUDA (только полная сборка), OpenSSL/libcurl (PoolClient), сборка через Makefile (g++-9 + nvcc) / Docker / Visual Studio.

---

## Среда верификации (важно)

В dev-окружении **нет компилятора** (ни `cl`/VS, ни `g++`, ни `nvcc`). Docker установлен, но демон может быть остановлен — его нужно запустить. Поэтому:

- **Юнит-тесты без GPU** (Tasks 1–2: `Checkpoint`, `RangeMath`) собираются и запускаются в контейнере `gcc:9` — это реальный red/green-цикл. Шаблон команды (из корня репозитория, PowerShell, Docker запущен):
  ```powershell
  docker run --rm -v "${PWD}:/src" -w /src gcc:9 sh -c "<build-and-run>"
  ```
- **Интеграционные задачи** (Tasks 3–6) затрагивают CUDA-смежный код — полностью собираются только в окружении пользователя (`make all` на GPU-машине или `docker build` по `Dockerfile`). Где возможно, делаем `-fsyntax-only` проверку не-CUDA файлов в контейнере.
- **Полный GPU-прогон** (interrupt/resume на puzzle 38) — на стороне пользователя (Task 7).

Если Docker не запущен — тесты Tasks 1–2 выполняет пользователь у себя; код в плане полный, чтобы это было возможно.

---

## File Structure

| Файл | Ответственность |
|------|------------------|
| `Pool/Checkpoint.h` / `.cpp` (новый) | Структура `Checkpoint` + атомарная запись/чтение/удаление файла. Чистый std C++. |
| `RangeMath.h` / `.cpp` (новый) | Чистая арифметика партиций над `Int`: ширина подотрезка, старт потока, offset из счётчика. |
| `tests/test_checkpoint.cpp` (новый) | Юнит-тесты `Checkpoint` (round-trip, отсутствие файла, битый файл, атомарность). |
| `tests/test_rangemath.cpp` (новый) | Юнит-тесты `RangeMath` (формула ширины, старты, покрытие без дыр, floor offset). |
| `Pool/PoolConfig.h` / `.cpp` | Флаг `saveProgress` + парсинг `save_progress`. |
| `Pool/PoolClient.h` / `.cpp` | `std::mutex` вокруг `foundKeys`; аксессор `getFoundProofPairs`. |
| `Vanity.h` / `.cpp` | resume-offset, экспонирование `numThreadsGPU`, progress-callback + периодическое сохранение; использование `RangeMath`. |
| `main.cpp` | CLI-аргумент; путь/создание `progress/`; установка progress-callback; удаление чекпойнта при завершении диапазона; ветка resume до основного цикла. |
| `Makefile`, `VanitySearch.vcxproj(.filters)` | Компиляция новых `.cpp`. |
| `.gitignore`, `pool.conf`, `README.md`, `Dockerfile` | `/progress`, опция конфига, документация. |

---

## Task 1: Модуль `Checkpoint` (чистый, TDD)

**Files:**
- Create: `Pool/Checkpoint.h`
- Create: `Pool/Checkpoint.cpp`
- Test: `tests/test_checkpoint.cpp`

- [ ] **Step 1: Написать заголовок `Pool/Checkpoint.h`**

```cpp
// Checkpoint persistence for resumable scan progress (pure std C++, no GPU/CUDA).
#ifndef POOL_CHECKPOINT_H
#define POOL_CHECKPOINT_H

#include <string>
#include <vector>
#include <utility>

struct Checkpoint {
    int puzzle = 0;
    std::string hex;
    std::string rangeStart;
    std::string rangeEnd;
    std::string targetAddress;
    std::vector<std::string> proofAddresses;
    int numThreads = 0;
    std::string offsetHex;                                        // per-thread advance, hex
    std::vector<std::pair<std::string, std::string>> foundProof;  // (address, privKeyHex)
};

namespace CheckpointIO {
    // Atomic write (temp file + rename). Returns false on I/O error.
    bool save(const std::string& path, const Checkpoint& cp);
    // Returns true only if a well-formed checkpoint was read.
    bool load(const std::string& path, Checkpoint& out);
    // Remove checkpoint (and any leftover temp). No error if absent.
    void clear(const std::string& path);
}

#endif // POOL_CHECKPOINT_H
```

- [ ] **Step 2: Написать тест `tests/test_checkpoint.cpp`**

```cpp
#include "Checkpoint.h"
#include <cstdio>
#include <string>

static int failures = 0;
#define CHECK(cond) do { if(!(cond)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); failures++; } } while(0)

int main() {
    const std::string path = "/tmp/btc_cp_test.txt";
    CheckpointIO::clear(path);

    // round-trip with lists and pairs
    Checkpoint cp;
    cp.puzzle = 71;
    cp.hex = "2A";
    cp.rangeStart = "0000000000000001";
    cp.rangeEnd = "0000000000100000";
    cp.targetAddress = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU";
    cp.proofAddresses = {"1AddrOne", "1AddrTwo", "1AddrThree"};
    cp.numThreads = 131072;
    cp.offsetHex = "ABCDEF";
    cp.foundProof = { {"1AddrOne", "00FF11"}, {"1AddrTwo", "DEADBEEF"} };
    CHECK(CheckpointIO::save(path, cp));

    Checkpoint got;
    CHECK(CheckpointIO::load(path, got));
    CHECK(got.puzzle == 71);
    CHECK(got.hex == "2A");
    CHECK(got.rangeStart == cp.rangeStart);
    CHECK(got.rangeEnd == cp.rangeEnd);
    CHECK(got.targetAddress == cp.targetAddress);
    CHECK(got.numThreads == 131072);
    CHECK(got.offsetHex == "ABCDEF");
    CHECK(got.proofAddresses.size() == 3);
    CHECK(got.proofAddresses[2] == "1AddrThree");
    CHECK(got.foundProof.size() == 2);
    CHECK(got.foundProof[1].first == "1AddrTwo");
    CHECK(got.foundProof[1].second == "DEADBEEF");

    // no temp file left behind
    FILE* tmp = fopen((path + ".tmp").c_str(), "r");
    CHECK(tmp == NULL);
    if (tmp) fclose(tmp);

    // missing file -> false
    CheckpointIO::clear(path);
    Checkpoint none;
    CHECK(CheckpointIO::load(path, none) == false);

    // corrupted file (no header) -> false
    FILE* f = fopen(path.c_str(), "w");
    fputs("garbage line\nfoo=bar\n", f);
    fclose(f);
    Checkpoint bad;
    CHECK(CheckpointIO::load(path, bad) == false);
    CheckpointIO::clear(path);

    if (failures == 0) { printf("ALL CHECKPOINT TESTS PASSED\n"); return 0; }
    printf("%d CHECK(S) FAILED\n", failures);
    return 1;
}
```

- [ ] **Step 3: Запустить тест — убедиться, что НЕ компилится (нет реализации)**

Run (Docker запущен):
```powershell
docker run --rm -v "${PWD}:/src" -w /src gcc:9 sh -c "g++ -O2 -std=c++17 -IPool tests/test_checkpoint.cpp Pool/Checkpoint.cpp -o /tmp/tc && /tmp/tc"
```
Expected: ошибка линковки/компиляции (нет `Pool/Checkpoint.cpp`).

- [ ] **Step 4: Реализовать `Pool/Checkpoint.cpp`**

```cpp
#include "Checkpoint.h"
#include <fstream>
#include <cstdio>
#include <cstdlib>

static const char* CP_HEADER = "btcpuzzle-checkpoint v1";

bool CheckpointIO::save(const std::string& path, const Checkpoint& cp) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out.is_open()) return false;
        out << CP_HEADER << "\n";
        out << "puzzle=" << cp.puzzle << "\n";
        out << "hex=" << cp.hex << "\n";
        out << "rangeStart=" << cp.rangeStart << "\n";
        out << "rangeEnd=" << cp.rangeEnd << "\n";
        out << "target=" << cp.targetAddress << "\n";
        out << "numThreads=" << cp.numThreads << "\n";
        out << "offset=" << cp.offsetHex << "\n";
        for (const auto& a : cp.proofAddresses) out << "proof=" << a << "\n";
        for (const auto& p : cp.foundProof)     out << "found=" << p.first << " " << p.second << "\n";
        out.flush();
        if (!out.good()) { out.close(); std::remove(tmp.c_str()); return false; }
    }
    std::remove(path.c_str());  // Windows rename() fails if destination exists
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return false; }
    return true;
}

bool CheckpointIO::load(const std::string& path, Checkpoint& out) {
    std::ifstream in(path);
    if (!in.is_open()) return false;

    std::string line;
    if (!std::getline(in, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != CP_HEADER) return false;

    Checkpoint cp;
    bool havePuzzle = false, haveNumThreads = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);
        if (key == "puzzle")          { cp.puzzle = std::atoi(val.c_str()); havePuzzle = true; }
        else if (key == "hex")         cp.hex = val;
        else if (key == "rangeStart")  cp.rangeStart = val;
        else if (key == "rangeEnd")    cp.rangeEnd = val;
        else if (key == "target")      cp.targetAddress = val;
        else if (key == "numThreads") { cp.numThreads = std::atoi(val.c_str()); haveNumThreads = true; }
        else if (key == "offset")      cp.offsetHex = val;
        else if (key == "proof")       cp.proofAddresses.push_back(val);
        else if (key == "found") {
            size_t sp = val.find(' ');
            if (sp != std::string::npos)
                cp.foundProof.emplace_back(val.substr(0, sp), val.substr(sp + 1));
        }
    }
    if (!havePuzzle || !haveNumThreads) return false;
    out = cp;
    return true;
}

void CheckpointIO::clear(const std::string& path) {
    std::remove(path.c_str());
    std::remove((path + ".tmp").c_str());
}
```

- [ ] **Step 5: Запустить тест — убедиться, что проходит**

Run:
```powershell
docker run --rm -v "${PWD}:/src" -w /src gcc:9 sh -c "g++ -O2 -std=c++17 -IPool tests/test_checkpoint.cpp Pool/Checkpoint.cpp -o /tmp/tc && /tmp/tc"
```
Expected: `ALL CHECKPOINT TESTS PASSED`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add Pool/Checkpoint.h Pool/Checkpoint.cpp tests/test_checkpoint.cpp
git commit -m "feat: add Checkpoint serialization module with tests"
```

---

## Task 2: Модуль `RangeMath` (арифметика партиций, TDD)

**Files:**
- Create: `RangeMath.h`
- Create: `RangeMath.cpp`
- Test: `tests/test_rangemath.cpp`

- [ ] **Step 1: Написать `RangeMath.h`**

```cpp
// Pure partition arithmetic over Int (no GPU). Must stay in sync with
// VanitySearch::getGPUStartingKeysMT (Vanity.cpp).
#ifndef RANGEMATH_H
#define RANGEMATH_H

#include "Int.h"

namespace RangeMath {
    // Per-thread sub-range width: (end[+1 if odd] - start) / numThreads.
    Int perThreadDiff(Int& start, Int& end, int numThreads);
    // Start key of thread idx: start + diff*idx.
    Int threadStart(Int& start, Int& diff, int idx);
    // Per-thread scan advance derived from total device key count: count / numThreads.
    Int offsetFromCount(uint64_t totalCount, int numThreads);
}

#endif // RANGEMATH_H
```

- [ ] **Step 2: Написать тест `tests/test_rangemath.cpp`**

```cpp
#include "RangeMath.h"
#include <cstdio>

static int failures = 0;
#define CHECK(cond) do { if(!(cond)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); failures++; } } while(0)

static bool eq(Int a, Int b) { return a.IsEqual(&b); }  // by value -> safe to call non-const

int main() {
    // start=0x100, end=0x140, N=4 -> diff=0x10
    Int start; start.SetBase16((char*)"100");
    Int end;   end.SetBase16((char*)"140");
    Int diff = RangeMath::perThreadDiff(start, end, 4);
    Int expDiff; expDiff.SetBase16((char*)"10");
    CHECK(eq(diff, expDiff));

    // thread starts: 0x100, 0x110, 0x120, 0x130
    const char* exp[4] = { "100", "110", "120", "130" };
    for (int i = 0; i < 4; i++) {
        Int s = RangeMath::threadStart(start, diff, i);
        Int e; e.SetBase16((char*)exp[i]);
        CHECK(eq(s, e));
    }

    // coverage with offset=0x6: [s_i, s_i+6) U [s_i+6, s_{i+1}) tiles range, no gaps
    Int offset; offset.SetBase16((char*)"6");
    for (int i = 0; i < 4; i++) {
        Int s     = RangeMath::threadStart(start, diff, i);
        Int sNext = RangeMath::threadStart(start, diff, i + 1);
        Int mid(&s); mid.Add(&offset);          // s_i + offset
        CHECK(mid.IsGreater(&s));                // pre-crash part non-empty
        CHECK(sNext.IsGreater(&mid));            // post part non-empty (offset < diff)
        Int chk(&s); chk.Add(&diff);            // s_i + diff == s_{i+1} (contiguous)
        CHECK(eq(chk, sNext));
    }

    // offsetFromCount: 0x40 keys over N=4 -> 0x10 per thread; floors 0x41 -> 0x10
    Int e10; e10.SetBase16((char*)"10");
    CHECK(eq(RangeMath::offsetFromCount(0x40, 4), e10));
    CHECK(eq(RangeMath::offsetFromCount(0x41, 4), e10));

    if (failures == 0) { printf("ALL RANGEMATH TESTS PASSED\n"); return 0; }
    printf("%d CHECK(S) FAILED\n", failures);
    return 1;
}
```

- [ ] **Step 3: Запустить тест — убедиться, что НЕ компилится**

Run:
```powershell
docker run --rm -v "${PWD}:/src" -w /src gcc:9 sh -c "g++ -O2 -std=c++17 -mssse3 -Wno-write-strings -I. tests/test_rangemath.cpp RangeMath.cpp Int.cpp IntMod.cpp IntGroup.cpp Random.cpp Timer.cpp -lpthread -o /tmp/trm && /tmp/trm"
```
Expected: ошибка (нет `RangeMath.cpp`).

- [ ] **Step 4: Реализовать `RangeMath.cpp`**

```cpp
#include "RangeMath.h"

namespace RangeMath {

Int perThreadDiff(Int& start, Int& end, int numThreads) {
    Int diff(&end);
    if (diff.IsOdd()) diff.AddOne();
    diff.Sub(&start);
    Int n; n.SetInt32((uint32_t)numThreads);
    diff.Div(&n);
    return diff;
}

Int threadStart(Int& start, Int& diff, int idx) {
    Int d(&diff);
    Int k; k.SetInt32((uint32_t)idx);
    d.Mult(&k);          // d = diff * idx
    Int s(&start);
    s.Add(&d);           // s = start + diff*idx
    return s;
}

Int offsetFromCount(uint64_t totalCount, int numThreads) {
    Int c(totalCount);   // Int(uint64_t)
    Int n; n.SetInt32((uint32_t)numThreads);
    c.Div(&n);
    return c;
}

} // namespace RangeMath
```

- [ ] **Step 5: Запустить тест — убедиться, что проходит**

Run:
```powershell
docker run --rm -v "${PWD}:/src" -w /src gcc:9 sh -c "g++ -O2 -std=c++17 -mssse3 -Wno-write-strings -I. tests/test_rangemath.cpp RangeMath.cpp Int.cpp IntMod.cpp IntGroup.cpp Random.cpp Timer.cpp -lpthread -o /tmp/trm && /tmp/trm"
```
Expected: `ALL RANGEMATH TESTS PASSED`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add RangeMath.h RangeMath.cpp tests/test_rangemath.cpp
git commit -m "feat: add RangeMath partition arithmetic with tests"
```

---

## Task 3: Флаг `save_progress` в конфиге и CLI

**Files:**
- Modify: `Pool/PoolConfig.h` (после `customRange`/`securityHash`, ~строки 41–42)
- Modify: `Pool/PoolConfig.cpp:105` (после парсинга `save_key`)
- Modify: `main.cpp` (блок аргументов, после `-savekey`, ~строки 601–609)
- Modify: `Pool/PoolClient.cpp` (вывод в `init()`, после блока `save_key`, ~строки 95–100)
- Modify: `pool.conf`

- [ ] **Step 1: Добавить поле в `Pool/PoolConfig.h`**

Найти:
```cpp
    // Other Settings
    std::string customRange = "none";
    std::string securityHash = "";
```
Заменить на:
```cpp
    // Other Settings
    std::string customRange = "none";
    std::string securityHash = "";

    // Save/resume scan progress to disk (resume an interrupted range on restart)
    bool saveProgress = true;
```

- [ ] **Step 2: Парсинг в `Pool/PoolConfig.cpp`**

Найти:
```cpp
        else if (key == "save_key") config.saveKeyToBtcPuzzle = parseBool(value);
```
Заменить на:
```cpp
        else if (key == "save_key") config.saveKeyToBtcPuzzle = parseBool(value);
        else if (key == "save_progress") config.saveProgress = parseBool(value);
```

- [ ] **Step 3: CLI-аргумент в `main.cpp`**

Найти блок (после `-savekey`):
```cpp
		else if (strcmp(argv[a], "-pubkey") == 0) {
```
Вставить ПЕРЕД ним:
```cpp
		else if (strcmp(argv[a], "-saveprogress") == 0) {
			if (a + 1 >= argc || (argv[a + 1][0] == '-' && argv[a + 1][1] != '\0')) {
				fprintf(stderr, "-saveprogress requires a value\n");
				exit(-1);
			}
			a++;
			g_poolConfig.saveProgress = (strcmp(argv[a], "true") == 0 || strcmp(argv[a], "1") == 0);
			a++;
		}
```

- [ ] **Step 4: Вывод состояния в `PoolClient::init()`**

Найти:
```cpp
	if (config.saveKeyToBtcPuzzle) {
		logMessage(INFO, "Save Key To Account  => Enabled");
	}
	else {
		logMessage(WARNING, "Save Key To Account  => Disabled");
	}
	std::cout << "========================================" << std::endl;
```
Заменить на:
```cpp
	if (config.saveKeyToBtcPuzzle) {
		logMessage(INFO, "Save Key To Account  => Enabled");
	}
	else {
		logMessage(WARNING, "Save Key To Account  => Disabled");
	}
	if (config.saveProgress) {
		logMessage(INFO, "Save Progress        => Enabled");
	}
	else {
		logMessage(WARNING, "Save Progress        => Disabled");
	}
	std::cout << "========================================" << std::endl;
```

- [ ] **Step 5: Добавить опцию в `pool.conf`**

Найти:
```
# Other Settings
custom_range=none
```
Заменить на:
```
# Other Settings
custom_range=none

# Save/resume scan progress (resume an interrupted range on restart)
save_progress=true
```

- [ ] **Step 6: Compile-check не-CUDA файлов (опционально, Docker)**

Run:
```powershell
docker run --rm -v "${PWD}:/src" -w /src gcc:9 sh -c "apt-get update -qq && apt-get install -y -qq libssl-dev >/dev/null 2>&1; g++ -fsyntax-only -std=c++17 -I. -IPool Pool/PoolConfig.cpp"
```
Expected: без ошибок (пустой вывод, exit 0).

- [ ] **Step 7: Commit**

```bash
git add Pool/PoolConfig.h Pool/PoolConfig.cpp Pool/PoolClient.cpp main.cpp pool.conf
git commit -m "feat: add save_progress config flag and CLI argument"
```

---

## Task 4: Потокобезопасный доступ к `foundKeys` + аксессор пар

**Files:**
- Modify: `Pool/PoolClient.h` (включить `<mutex>`, `<utility>`; добавить мьютекс и метод)
- Modify: `Pool/PoolClient.cpp` (`onKeyFound`, `hasAllProofKeys`, `getProofKeys` + новый `getFoundProofPairs`)

- [ ] **Step 1: Объявления в `Pool/PoolClient.h`**

Найти:
```cpp
#include <thread>
#include <atomic>
#include <chrono>
```
Заменить на:
```cpp
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <utility>
```

Найти:
```cpp
    std::map<std::string, FoundKey> foundKeys;
    int rangesScanned;
```
Заменить на:
```cpp
    std::map<std::string, FoundKey> foundKeys;
    std::mutex foundKeysMutex;
    int rangesScanned;
```

Найти:
```cpp
    // Get list of found proof keys
    std::vector<std::string> getProofKeys(const RangeData& range);
```
Заменить на:
```cpp
    // Get list of found proof keys
    std::vector<std::string> getProofKeys(const RangeData& range);
    // (address, privKeyHex) pairs for proof addresses found so far (for checkpointing)
    std::vector<std::pair<std::string, std::string>> getFoundProofPairs(const RangeData& range);
```

- [ ] **Step 2: Блокировки в `Pool/PoolClient.cpp::onKeyFound`**

Найти:
```cpp
void PoolClient::onKeyFound(const std::string& address, const std::string& privateKey) {
	FoundKey fk;
	fk.address = address;
	fk.privateKey = privateKey;
	fk.isProof = false;
	fk.isReward = false;
	fk.isTarget = false;

	foundKeys[address] = fk;
	keysFound++;
```
Заменить на:
```cpp
void PoolClient::onKeyFound(const std::string& address, const std::string& privateKey) {
	FoundKey fk;
	fk.address = address;
	fk.privateKey = privateKey;
	fk.isProof = false;
	fk.isReward = false;
	fk.isTarget = false;

	{
		std::lock_guard<std::mutex> lk(foundKeysMutex);
		foundKeys[address] = fk;
		keysFound++;
	}
```

- [ ] **Step 3: Блокировки в `hasAllProofKeys` / `getProofKeys` и новый метод**

Найти:
```cpp
bool PoolClient::hasAllProofKeys(const RangeData& range) {
	int found = 0;
	for (const auto& addr : range.proofOfWorkAddresses) {
		if (foundKeys.find(addr) != foundKeys.end()) {
			found++;
		}
	}
	return found == (int)range.proofOfWorkAddresses.size();
}

std::vector<std::string> PoolClient::getProofKeys(const RangeData& range) {
	std::vector<std::string> proofKeys;
	for (const auto& addr : range.proofOfWorkAddresses) {
		if (foundKeys.find(addr) != foundKeys.end()) {
			proofKeys.push_back(foundKeys[addr].privateKey);
		}
	}
	return proofKeys;
}
```
Заменить на:
```cpp
bool PoolClient::hasAllProofKeys(const RangeData& range) {
	std::lock_guard<std::mutex> lk(foundKeysMutex);
	int found = 0;
	for (const auto& addr : range.proofOfWorkAddresses) {
		if (foundKeys.find(addr) != foundKeys.end()) {
			found++;
		}
	}
	return found == (int)range.proofOfWorkAddresses.size();
}

std::vector<std::string> PoolClient::getProofKeys(const RangeData& range) {
	std::lock_guard<std::mutex> lk(foundKeysMutex);
	std::vector<std::string> proofKeys;
	for (const auto& addr : range.proofOfWorkAddresses) {
		if (foundKeys.find(addr) != foundKeys.end()) {
			proofKeys.push_back(foundKeys[addr].privateKey);
		}
	}
	return proofKeys;
}

std::vector<std::pair<std::string, std::string>> PoolClient::getFoundProofPairs(const RangeData& range) {
	std::lock_guard<std::mutex> lk(foundKeysMutex);
	std::vector<std::pair<std::string, std::string>> pairs;
	for (const auto& addr : range.proofOfWorkAddresses) {
		auto it = foundKeys.find(addr);
		if (it != foundKeys.end()) {
			pairs.emplace_back(addr, it->second.privateKey);
		}
	}
	return pairs;
}
```

> Примечание: `reset()` вызывается только между диапазонами в главном потоке, когда GPU-поток уже завершён, поэтому отдельная блокировка там не требуется. При желании можно тоже обернуть `foundKeys.clear()` в `lock_guard` — безвредно.

- [ ] **Step 4: Compile-check (опционально, Docker)**

Run:
```powershell
docker run --rm -v "${PWD}:/src" -w /src gcc:9 sh -c "apt-get update -qq && apt-get install -y -qq libssl-dev libcurl4-openssl-dev >/dev/null 2>&1; g++ -fsyntax-only -std=c++17 -I. -IPool Pool/PoolClient.cpp"
```
Expected: без ошибок (exit 0).

- [ ] **Step 5: Commit**

```bash
git add Pool/PoolClient.h Pool/PoolClient.cpp
git commit -m "feat: guard foundKeys with a mutex; add getFoundProofPairs"
```

---

## Task 5: `VanitySearch` — resume-offset, progress-callback, периодическое сохранение

**Files:**
- Modify: `Vanity.h` (callback-тип, сеттеры, члены)
- Modify: `Vanity.cpp` (include, конструктор, `getGPUStartingKeysMT`, `FindKeyGPU`, `Search`)

- [ ] **Step 1: Объявления в `Vanity.h`**

Найти:
```cpp
	void Search(std::vector<int> gpuId, std::vector<int> gridSize);
	void FindKeyGPU(TH_PARAM* p);
	void setKeyFoundCallback(KeyFoundCallback cb) {
		keyCallback = cb;
	}
```
Заменить на:
```cpp
	void Search(std::vector<int> gpuId, std::vector<int> gridSize);
	void FindKeyGPU(TH_PARAM* p);
	void setKeyFoundCallback(KeyFoundCallback cb) {
		keyCallback = cb;
	}

	// Progress persistence (pool save/resume)
	typedef std::function<void(Int&, int)> ProgressCallback;
	void setProgressCallback(ProgressCallback cb) { progressCallback = cb; }
	// Resume an interrupted range: shift each GPU thread by `offset` if the thread
	// count matches `expectedThreads`; otherwise FindKeyGPU falls back to a full re-scan.
	void setResume(Int& offset, int expectedThreads) {
		resumeOffset.Set(&offset);
		resumeExpectedThreads = expectedThreads;
	}
```

Найти:
```cpp
	KeyFoundCallback keyCallback;
	int searchType;
```
Заменить на:
```cpp
	KeyFoundCallback keyCallback;
	ProgressCallback progressCallback;
	Int resumeOffset;
	int resumeExpectedThreads;
	int savedNumThreads;
	int searchType;
```

- [ ] **Step 2: Include и инициализация в `Vanity.cpp`**

Найти (рядом с прочими include, верх файла):
```cpp
#include "Vanity.h"
```
Заменить на:
```cpp
#include "Vanity.h"
#include "RangeMath.h"
```

Найти (в конструкторе):
```cpp
	startKey.Set(&bc->ksNext);	
```
Заменить на:
```cpp
	startKey.Set(&bc->ksNext);	
	resumeOffset.SetInt32(0);
	resumeExpectedThreads = 0;
	savedNumThreads = 0;
```

- [ ] **Step 3: `getGPUStartingKeysMT` — использовать `RangeMath::perThreadDiff`**

Найти:
```cpp
	tRangeDiffPerGPUThread.Set(&tRangeEnd);
	if (tRangeDiffPerGPUThread.IsOdd())
	{
		tRangeDiffPerGPUThread.AddOne();
	}
	tRangeDiffPerGPUThread.Sub(&tRangeStart);

	Int tRangeDiffGlobal(&tRangeDiffPerGPUThread);

	// this only needed if total GPU threads not power of 2
	//tRangeDiffPerThread.Add(&numThreads);

	tRangeDiffPerGPUThread.Div(&tNumThreadsGPU);
```
Заменить на:
```cpp
	// Global span (end[+1 if odd] - start), used below for CPU-thread division.
	Int tRangeDiffGlobal(&tRangeEnd);
	if (tRangeDiffGlobal.IsOdd())
	{
		tRangeDiffGlobal.AddOne();
	}
	tRangeDiffGlobal.Sub(&tRangeStart);

	// Per-GPU-thread width (kept in sync with RangeMath, used by tests/resume).
	Int _ptd = RangeMath::perThreadDiff(tRangeStart, tRangeEnd, numThreadsGPU);
	tRangeDiffPerGPUThread.Set(&_ptd);
```

> `tNumThreadsGPU` после этого может стать неиспользуемой (предупреждение, не ошибка) — оставить как есть.

- [ ] **Step 4: `FindKeyGPU` — сохранить число потоков и применить resume-offset**

Найти:
```cpp
	int thId = ph->threadId;
	GPUEngine g(ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound);
	int numThreadsGPU = g.GetNumThreadsGPU();
```
Заменить на:
```cpp
	int thId = ph->threadId;
	GPUEngine g(ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound);
	int numThreadsGPU = g.GetNumThreadsGPU();
	savedNumThreads = numThreadsGPU;
```

Найти:
```cpp
	getGPUStartingKeysMT(bc->ksStart, bc->ksFinish, g.GetGroupSize(), numThreadsGPU, privateKeys, publicKeys);
	
	// copy to gpu
	ok = g.SetKeys(publicKeys);
```
Заменить на:
```cpp
	getGPUStartingKeysMT(bc->ksStart, bc->ksFinish, g.GetGroupSize(), numThreadsGPU, privateKeys, publicKeys);

	// Resume: shift each thread's start key by the saved per-thread offset.
	if (!resumeOffset.IsZero()) {
		Int effOffset(&resumeOffset);
		if (resumeExpectedThreads != numThreadsGPU) {
			fprintf(stdout, "[RESUME] GPU thread count changed (%d -> %d): full range re-scan\n",
				resumeExpectedThreads, numThreadsGPU);
			fflush(stdout);
			effOffset.SetInt32(0);
		}
		if (!effOffset.IsZero()) {
			int gs = g.GetGroupSize();
			for (int i = 0; i < numThreadsGPU; i++) {
				privateKeys[i].Add(&effOffset);
				Int pk(&privateKeys[i]);
				pk.Add((uint64_t)(gs / 2));
				publicKeys[i] = secp->ComputePublicKey(&pk);
			}
			fprintf(stdout, "[RESUME] resumed from per-thread offset 0x%s\n", effOffset.GetBase16().c_str());
			fflush(stdout);
		}
	}

	// copy to gpu
	ok = g.SetKeys(publicKeys);
```

- [ ] **Step 5: `Search` — периодическое сохранение прогресса**

Найти:
```cpp
			timeout60sec += (t1 - t0);
			if (timeout60sec > 2.0) {	

				// Save LowerPrivKey as saveProgress
				/*saveProgress(params, lastSaveKey, bc);

				// Reached end of keyspace
				if (lastSaveKey.IsGreaterOrEqual(&bc->ksFinish)) {
					endOfSearch = true;
					fprintf(stdout, "[EXIT] Range search completed \n");	
					fflush(stdout);
				}*/

				timeout60sec = 0.0;
			}
```
Заменить на:
```cpp
			timeout60sec += (t1 - t0);
			if (timeout60sec > 5.0) {

				// Persist scan progress for crash-safe resume.
				if (progressCallback && savedNumThreads > 0) {
					Int off = RangeMath::offsetFromCount(getGPUCount(), savedNumThreads);
					progressCallback(off, savedNumThreads);
				}

				timeout60sec = 0.0;
			}
```

> Достаточно периодического сохранения: `offset` всегда консервативно «позади» — всё, что найдено после последнего снапшота, при resume попадёт в зону повторного скана и будет найдено снова. Отдельное сохранение «на каждый найденный ключ» не требуется для корректности.

- [ ] **Step 6: Удалить мёртвый `saveProgress` (необязательная чистка)**

Удалить объявление в `Vanity.h`:
```cpp
	void saveProgress(TH_PARAM* p, Int& lastSaveKey, BITCRACK_PARAM* bc);
```
Удалить определение в `Vanity.cpp`:
```cpp
void VanitySearch::saveProgress(TH_PARAM* p, Int& lastSaveKey, BITCRACK_PARAM* bc) {

	Int lowerKey;
	lowerKey.Set(&p[0].THnextKey);

	int total = numGPUs;
	for (int i = 0; i < total; i++) {
		if (p[i].THnextKey.IsLower(&lowerKey))
			lowerKey.Set(&p[i].THnextKey);
	}

	if (lowerKey.IsLowerOrEqual(&lastSaveKey)) return;
	lastSaveKey.Set(&lowerKey);
}
```
Удалить в `Search` неиспользуемые:
```cpp
	Int lastSaveKey; 
	lastSaveKey.SetInt32(0);
```

- [ ] **Step 7: Сборка (на стороне пользователя / полный билд)**

Это CUDA-смежный код — компилируется только в полном окружении. Проверяется в Task 7 общей сборкой.

- [ ] **Step 8: Commit**

```bash
git add Vanity.h Vanity.cpp
git commit -m "feat: VanitySearch resume offset + periodic progress callback"
```

---

## Task 6: Интеграция в `main.cpp` (запись чекпойнта, ветка resume)

**Files:**
- Modify: `main.cpp` (include; путь+каталог `progress/`; ветка resume; progress-callback; удаление чекпойнта)

- [ ] **Step 1: Include модуля Checkpoint**

Найти:
```cpp
#include "Pool/PoolClient.h"
```
Заменить на:
```cpp
#include "Pool/PoolClient.h"
#include "Pool/Checkpoint.h"
```

- [ ] **Step 2: Путь чекпойнта и каталог `progress/` + ветка resume**

Найти:
```cpp
	int loops = 0;
	while (true)
	{
```
Заменить на (вставив весь блок ПЕРЕД `int loops = 0;`):
```cpp
	// Checkpoint path (one per GPU instance) + ensure progress dir exists.
	std::string cpPath = "progress/checkpoint_gpu" + std::to_string(g_poolConfig.gpuIndex) + ".txt";
	{
		struct stat stp = { 0 };
		if (stat("progress", &stp) == -1) {
#ifdef _WIN32
			_mkdir("progress");
#else
			mkdir("progress", 0755);
#endif
		}
	}

	// ============ RESUME: finish an interrupted range, then submit ============
	if (g_poolMode && g_poolConfig.saveProgress) {
		Checkpoint cp;
		if (CheckpointIO::load(cpPath, cp)) {
			bool stale = (cp.puzzle != g_poolConfig.targetPuzzle) || cp.hex.empty()
				|| cp.targetAddress.empty() || cp.rangeStart.empty() || cp.rangeEnd.empty();
			if (stale) {
				logMessage(WARNING, "[RESUME] Stale/foreign checkpoint, discarding.");
				CheckpointIO::clear(cpPath);
			}
			else {
				logMessage(WARNING, "[RESUME] Unfinished range found; finishing before requesting a new one.");

				RangeData rdata;
				rdata.hex = cp.hex;
				rdata.rangeStart = cp.rangeStart;
				rdata.rangeEnd = cp.rangeEnd;
				rdata.targetAddress = cp.targetAddress;
				rdata.proofOfWorkAddresses = cp.proofAddresses;
				rdata.success = true;

				PoolClient rclient(g_poolConfig);
				if (!rclient.init()) {
					logMessage(DANGER, "[RESUME] Pool client init failed; skipping resume.");
					CheckpointIO::clear(cpPath);
				}
				else {
					// Replay proof keys found before the crash.
					for (const auto& pr : cp.foundProof) {
						rclient.onKeyFound(pr.first, pr.second);
					}

					// Rebuild keyspace.
					getKeySpace(cp.hex + cp.rangeStart + ":+" + cp.rangeEnd, bc, maxKey);
					bc->ksNext.Set(&bc->ksStart);
					checkKeySpace(bc, maxKey);

					// Recreate address file and parse.
					struct stat st = { 0 };
					if (stat("ranges", &st) == -1) {
#ifdef _WIN32
						_mkdir("ranges");
#else
						mkdir("ranges", 0755);
#endif
					}
					std::string maskedHex = cp.hex;
					if (maskedHex.length() >= 2 && g_poolConfig.untrustedComputer) {
						maskedHex[1] = '_';
						maskedHex[maskedHex.length() - 1] = '_';
					}
					std::string rfile = "ranges/btcpuzzle_" + maskedHex + ".txt";
					{
						std::ofstream out(rfile);
						for (const auto& a : cp.proofAddresses) out << a << "\n";
						out << cp.targetAddress << "\n";
					}
					address.clear();
					parseFile(rfile, address);
					if (g_poolConfig.untrustedComputer) std::remove(rfile.c_str());

					VanitySearch* rv = new VanitySearch(secp, address, searchMode, stop, outputFile, maxFound, bc);
					rclient.startPing(rdata.hex);
					rv->setKeyFoundCallback([&](std::string addr, std::string key) {
						while (key.length() < 64) key = "0" + key;
						rclient.onKeyFound(addr, key);
						if (addr == rdata.targetAddress) {
							printf("\n*** TARGET KEY FOUND (during resume)! ***\n");
							rclient.notifyTargetFound(addr, key);
							std::string fn = "WINNER_" + addr.substr(0, 8) + ".txt";
							std::ofstream of(fn);
							if (of.is_open()) {
								of << "Target Address: " << addr << "\n\n";
								if (g_poolConfig.untrustedComputer)
									of << "Private Key: " << rclient.encryptData(key) << "\n";
								else
									of << "Private Key: " << key << "\n";
							}
						}
					});

					Int rOff; rOff.SetInt32(0);
					if (!cp.offsetHex.empty()) rOff.SetBase16((char*)cp.offsetHex.c_str());
					rv->setResume(rOff, cp.numThreads);

					rv->Search(gpuId, gridSize);
					rclient.stopPing();

					if (rclient.hasAllProofKeys(rdata)) {
						auto proofKeys = rclient.getProofKeys(rdata);
						bool done = false;
						for (int attempt = 1; attempt <= 3 && !done; attempt++) {
							if (rclient.submitRange(rdata.hex, proofKeys)) {
								rclient.notifyRangeScanned(rdata.hex);
								logMessage(SUCCESS, "[RESUME] Finished range submitted.");
								done = true;
							}
							else {
								logMessage(DANGER, "[RESUME] Submit failed, retrying...");
								std::this_thread::sleep_for(std::chrono::seconds(5));
							}
						}
						if (!done) {
							std::string errFile = "FLAG_ERROR_" + rdata.hex + ".txt";
							std::ofstream eo(errFile);
							if (eo.is_open()) {
								eo << "Range: " << rdata.hex << "\n";
								for (auto& k : proofKeys) eo << k << "\n";
							}
							logMessage(WARNING, "[RESUME] Pool did not accept the range; proof keys backed up.");
						}
					}
					else {
						logMessage(DANGER, "[RESUME] Not all proof keys after re-scan; discarding.");
					}

					delete rv;
					rclient.reset();
					CheckpointIO::clear(cpPath);
				}
			}
		}
	}

	int loops = 0;
	while (true)
	{
```

- [ ] **Step 3: Установить progress-callback в основном цикле**

Найти (конец блока установки keyFoundCallback):
```cpp
				});
		}

		// Start search
		v->Search(gpuId, gridSize);
```
Заменить на:
```cpp
				});

			if (g_poolConfig.saveProgress) {
				v->setProgressCallback([&](Int& off, int n) {
					Checkpoint cp;
					cp.puzzle = g_poolConfig.targetPuzzle;
					cp.hex = rangeData.hex;
					cp.rangeStart = rangeData.rangeStart;
					cp.rangeEnd = rangeData.rangeEnd;
					cp.targetAddress = rangeData.targetAddress;
					cp.proofAddresses = rangeData.proofOfWorkAddresses;
					cp.numThreads = n;
					cp.offsetHex = off.GetBase16();
					cp.foundProof = client.getFoundProofPairs(rangeData);
					CheckpointIO::save(cpPath, cp);
				});
			}
		}

		// Start search
		v->Search(gpuId, gridSize);
```

- [ ] **Step 4: Удалять чекпойнт при штатном завершении диапазона**

Найти:
```cpp
			// Clean up
			client.reset();
			loops++;
```
Заменить на:
```cpp
			// Clean up
			if (g_poolConfig.saveProgress) CheckpointIO::clear(cpPath);
			client.reset();
			loops++;
```

- [ ] **Step 5: Сборка — полное окружение (Task 7).**

- [ ] **Step 6: Commit**

```bash
git add main.cpp
git commit -m "feat: write progress checkpoint and resume interrupted range on startup"
```

---

## Task 7: Сборка, build-файлы, документация, GPU smoke-test

**Files:**
- Modify: `Makefile` (новые `.cpp`/`.o`)
- Modify: `VanitySearch.vcxproj`, `VanitySearch.vcxproj.filters` (новые `ClCompile`)
- Modify: `.gitignore` (`/progress`)
- Modify: `README.md`, `Dockerfile`, `btcpuzzle.sh`

- [ ] **Step 1: Makefile — добавить исходники**

Найти:
```make
SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp Bech32.cpp Wildcard.cpp \
      Pool/PoolConfig.cpp Pool/PoolClient.cpp Pool/Logger.cpp
```
Заменить на:
```make
SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp RangeMath.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp Bech32.cpp Wildcard.cpp \
      Pool/PoolConfig.cpp Pool/PoolClient.cpp Pool/Logger.cpp Pool/Checkpoint.cpp
```

В обоих блоках `OBJET` (gpu и не-gpu) найти:
```make
        Vanity.o GPU/GPUGenerate.o \
```
Заменить на:
```make
        Vanity.o RangeMath.o GPU/GPUGenerate.o \
```

И в обоих блоках `OBJET` найти:
```make
        Pool/PoolConfig.o Pool/PoolClient.o Pool/Logger.o)
```
Заменить на:
```make
        Pool/PoolConfig.o Pool/PoolClient.o Pool/Logger.o Pool/Checkpoint.o)
```

- [ ] **Step 2: Visual Studio project — добавить `ClCompile`**

В `VanitySearch.vcxproj` найти строку с `<ClCompile Include="Vanity.cpp" />` и добавить рядом:
```xml
    <ClCompile Include="RangeMath.cpp" />
    <ClCompile Include="Pool\Checkpoint.cpp" />
```
В `VanitySearch.vcxproj.filters` добавить аналогичные `<ClCompile Include="RangeMath.cpp">` (фильтр `Source Files`) и `<ClCompile Include="Pool\Checkpoint.cpp">` (фильтр `Source Files\Pool` если есть, иначе `Source Files`). При желании добавить `<ClInclude>` для `RangeMath.h`, `Pool\Checkpoint.h`.

- [ ] **Step 3: `.gitignore` — игнорировать `progress/`**

Найти:
```
/ranges
```
Заменить на:
```
/ranges
/progress
```

- [ ] **Step 4: README — описать опцию**

Добавить в раздел `# Arguments` строку:
```
-saveprogress => Save/resume scan progress (true|false, default true). On restart the client finishes the interrupted range and submits it before requesting a new one.
```
И в `# Default Settings` (пример `pool.conf`) после `custom_range=none` добавить:
```
# Save/resume scan progress (resume an interrupted range on restart)
save_progress=true
```

- [ ] **Step 5: Dockerfile + btcpuzzle.sh — пробросить опцию через ENV**

В `Dockerfile` найти:
```dockerfile
ENV SAVE_KEY="false"
```
Заменить на:
```dockerfile
ENV SAVE_KEY="false"
ENV SAVE_PROGRESS="true"
```

В `btcpuzzle.sh` найти:
```bash
  [ -n "$SAVE_KEY" ] && [ "$SAVE_KEY" != "false" ] && CMD+=("-savekey" "$SAVE_KEY")
```
Заменить на:
```bash
  [ -n "$SAVE_KEY" ] && [ "$SAVE_KEY" != "false" ] && CMD+=("-savekey" "$SAVE_KEY")
  [ -n "$SAVE_PROGRESS" ] && CMD+=("-saveprogress" "$SAVE_PROGRESS")
```

> Примечание: `btcpuzzle.sh` запускает по процессу на GPU с `-gpuId $i`, поэтому путь чекпойнта `progress/checkpoint_gpu<idx>.txt` корректно разделяет инстансы при мульти-GPU.

- [ ] **Step 6: Полная сборка (окружение пользователя)**

Linux/GPU-машина:
```bash
make clean && make all
```
Expected: бинарь `vanitysearch` собран без ошибок.

Либо Docker-образ:
```bash
docker build --build-arg BUILD_HASH="<hash>" -t btcpuzzle .
```
Expected: образ собран.

- [ ] **Step 7: GPU smoke-test (interrupt → resume)**

1. Запустить на puzzle 38 (быстрый), дать просканировать часть диапазона, увидеть в `progress/checkpoint_gpu0.txt` непустой `offset`.
2. Прервать процесс (Ctrl+C / kill).
3. Перезапустить. Ожидать в логах:
   `[RESUME] Unfinished range found; finishing before requesting a new one.`
   и `[RESUME] resumed from per-thread offset 0x...`.
4. Убедиться, что диапазон досканирован, proof-ключи засабмичены (`[RESUME] Finished range submitted.`), чекпойнт удалён, клиент перешёл к новому диапазону.
5. Контроль смены GPU-геометрии: перезапустить с другим `-g` grid → ожидать
   `[RESUME] GPU thread count changed (... -> ...): full range re-scan` и корректное завершение.

- [ ] **Step 8: Commit**

```bash
git add Makefile VanitySearch.vcxproj VanitySearch.vcxproj.filters .gitignore README.md Dockerfile btcpuzzle.sh
git commit -m "build/docs: compile new sources, ignore progress/, document save_progress"
```

---

## Self-Review

**1. Spec coverage:**
- §4.1 offset = counters/N → RangeMath::offsetFromCount + Search (Task 2, 5). ✓
- §4.2 формат/поля/атомарность → Checkpoint (Task 1). ✓
- §4.3 периодическое сохранение → Search timer (Task 5). Упрощение «без отдельного сохранения на каждый ключ» обосновано (offset консервативен). ✓
- §4.4 resume точный/fallback → FindKeyGPU + main resume branch (Task 5, 6). ✓
- §4.5 интеграция (perThreadDiff, callback) → Task 5. ✓
- §4.55 мьютекс foundKeys → Task 4. ✓
- §4.6 bounded submit → main resume branch (3 попытки + FLAG_ERROR). ✓
- §4.7 конфиг-флаг → Task 3. ✓
- §5 точки изменений → Tasks 3–7 + build-файлы. ✓
- §6 граничные случаи (stale checkpoint, N mismatch, save_progress=false, битый файл) → Task 1/6. ✓
- §7 тестирование → Tasks 1–2 (контейнер) + Task 7 (GPU). ✓

**2. Placeholder scan:** Полный код во всех шагах; команды с ожидаемым выводом. Единственный частично-описательный шаг — вставка `ClCompile` в `.vcxproj` (Task 7.2): точка вставки — рядом с существующей строкой `Vanity.cpp`, XML тривиально локализуется. Прочих заглушек нет.

**3. Type consistency:**
- `Checkpoint` поля/типы идентичны в Task 1 и Task 6. ✓
- `getFoundProofPairs` сигнатура совпадает (Task 4 ↔ Task 6). ✓
- `setResume(Int&, int)`, `setProgressCallback(std::function<void(Int&,int)>)` — совпадают (Task 5 ↔ Task 6). ✓
- `RangeMath::perThreadDiff/threadStart/offsetFromCount` — совпадают (Task 2 ↔ Task 5). ✓
- `offsetHex` ↔ `Int::GetBase16()`/`SetBase16` — согласовано. ✓

---

## Открытые замечания (не блокеры)
- `submitRange()` возвращает `true` даже при HTTP≠200 (false только при пустом ответе) — bounded-retry в resume повторяет лишь сетевые сбои; явный отказ пула приводит к удалению чекпойнта и переходу дальше. Это совпадает с текущим поведением клиента.
- В основном цикле `parseFile()` дописывает в `address` без очистки (предсуществующее поведение) — в resume-ветке мы вызываем `address.clear()` явно. Возможный предсуществующий баг основного цикла вне рамок этой задачи.
