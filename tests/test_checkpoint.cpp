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
