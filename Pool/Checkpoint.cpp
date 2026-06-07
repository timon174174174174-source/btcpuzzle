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
