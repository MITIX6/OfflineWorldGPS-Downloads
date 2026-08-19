#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::size_t kHeaderSize = 64;
constexpr double kEarthRadius = 6371000.0;

std::uint32_t readU32(const std::uint8_t* data, std::size_t offset) {
    std::uint32_t value;
    std::memcpy(&value, data + offset, sizeof(value));
    return value;
}

double metres(double lat1, double lon1, double lat2, double lon2) {
    const double pi = std::acos(-1.0);
    double mean = (lat1 + lat2) * 0.5 * pi / 180.0;
    double x = (lon2 - lon1) * pi / 180.0 * std::cos(mean);
    double y = (lat2 - lat1) * pi / 180.0;
    return kEarthRadius * std::hypot(x, y);
}

struct QueueItem {
    float score;
    std::uint32_t node;
    bool operator<(const QueueItem& other) const { return score > other.score; }
};

class Graph {
public:
    explicit Graph(const std::string& path) {
        descriptor_ = open(path.c_str(), O_RDONLY);
        if (descriptor_ < 0) throw std::runtime_error("Cannot open graph");
        struct stat status {};
        if (fstat(descriptor_, &status) != 0) throw std::runtime_error("Cannot stat graph");
        size_ = static_cast<std::size_t>(status.st_size);
        data_ = static_cast<std::uint8_t*>(
            mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0));
        if (data_ == MAP_FAILED) throw std::runtime_error("Cannot map graph");
        if (size_ < kHeaderSize || std::memcmp(data_, "OWGRT001", 8) != 0) {
            throw std::runtime_error("Invalid OWGR graph");
        }
        nodes_ = readU32(data_, 12);
        edges_ = readU32(data_, 16);
        scale_ = readU32(data_, 20);
        spainNodes_ = readU32(data_, 24);
        franceNodes_ = readU32(data_, 28);
        switzerlandNodes_ = readU32(data_, 32);
        std::size_t expected = kHeaderSize + static_cast<std::size_t>(nodes_) * 12 + 4
            + static_cast<std::size_t>(edges_) * 12;
        if (scale_ != 1000000 || expected != size_) {
            throw std::runtime_error("Truncated or incompatible graph");
        }
        const std::uint8_t* cursor = data_ + kHeaderSize;
        latitudes_ = reinterpret_cast<const std::int32_t*>(cursor);
        cursor += static_cast<std::size_t>(nodes_) * 4;
        longitudes_ = reinterpret_cast<const std::int32_t*>(cursor);
        cursor += static_cast<std::size_t>(nodes_) * 4;
        offsets_ = reinterpret_cast<const std::uint32_t*>(cursor);
        cursor += static_cast<std::size_t>(nodes_ + 1) * 4;
        targets_ = reinterpret_cast<const std::uint32_t*>(cursor);
        cursor += static_cast<std::size_t>(edges_) * 4;
        distances_ = reinterpret_cast<const std::uint32_t*>(cursor);
        cursor += static_cast<std::size_t>(edges_) * 4;
        durations_ = reinterpret_cast<const std::uint32_t*>(cursor);
        if (offsets_[nodes_] != edges_) throw std::runtime_error("Invalid adjacency offsets");
    }

    ~Graph() {
        if (data_ && data_ != MAP_FAILED) munmap(data_, size_);
        if (descriptor_ >= 0) close(descriptor_);
    }

    std::uint32_t nearest(double latitude, double longitude) const {
        double best = std::numeric_limits<double>::infinity();
        std::uint32_t result = 0;
        for (std::uint32_t node = 0; node < nodes_; ++node) {
            if (offsets_[node + 1] - offsets_[node] < 2) continue;
            double distance = metres(latitude, longitude, lat(node), lon(node));
            if (distance < best) {
                best = distance;
                result = node;
            }
        }
        if (best > 5000.0) throw std::runtime_error("Test endpoint is too far from the graph");
        std::cout << "Snapped endpoint " << best << " m from node " << result << '\n';
        return result;
    }

    float heuristic(std::uint32_t node, std::uint32_t destination) const {
        return static_cast<float>(metres(lat(node), lon(node), lat(destination), lon(destination))
                                  / (130000.0 / 3600.0));
    }

    void testRoute(double startLat, double startLon, double endLat, double endLon) const {
        std::uint32_t start = nearest(startLat, startLon);
        std::uint32_t destination = nearest(endLat, endLon);
        std::vector<float> best(nodes_, std::numeric_limits<float>::infinity());
        std::vector<std::int32_t> parent(nodes_, -1);
        std::vector<std::uint8_t> closed(nodes_, 0);
        std::priority_queue<QueueItem> queue;
        best[start] = 0.0f;
        queue.push({heuristic(start, destination), start});
        std::uint64_t expanded = 0;
        while (!queue.empty()) {
            QueueItem item = queue.top();
            queue.pop();
            std::uint32_t node = item.node;
            if (closed[node]) continue;
            closed[node] = 1;
            if (node == destination) break;
            for (std::uint32_t edge = offsets_[node]; edge < offsets_[node + 1]; ++edge) {
                std::uint32_t target = targets_[edge];
                if (target >= nodes_) throw std::runtime_error("Invalid edge target");
                float candidate = best[node] + static_cast<float>(durations_[edge]);
                if (candidate < best[target]) {
                    best[target] = candidate;
                    parent[target] = static_cast<std::int32_t>(node);
                    queue.push({candidate + heuristic(target, destination), target});
                }
            }
            ++expanded;
            if (expanded % 5000000 == 0) {
                std::cout << "Expanded " << expanded << " nodes\n";
            }
        }
        if (!std::isfinite(best[destination])) throw std::runtime_error("No route found");
        bool sawSpain = false;
        bool sawFrance = false;
        bool sawSwitzerland = false;
        std::uint64_t pathNodes = 0;
        for (std::int32_t node = static_cast<std::int32_t>(destination); node >= 0;
             node = parent[node]) {
            ++pathNodes;
            std::uint32_t value = static_cast<std::uint32_t>(node);
            if (value < spainNodes_) sawSpain = true;
            else if (value < spainNodes_ + franceNodes_) sawFrance = true;
            else if (value < spainNodes_ + franceNodes_ + switzerlandNodes_) {
                sawSwitzerland = true;
            }
            if (value == start) break;
            if (pathNodes > nodes_) throw std::runtime_error("Parent cycle");
        }
        if (!sawSpain || !sawFrance || !sawSwitzerland) {
            throw std::runtime_error("Route does not traverse all three country graphs");
        }
        std::cout << "Route validated: " << pathNodes << " path nodes; " << expanded
                  << " expanded nodes; " << best[destination] / 3600.0f << " hours\n";
    }

private:
    double lat(std::uint32_t node) const { return latitudes_[node] / 1000000.0; }
    double lon(std::uint32_t node) const { return longitudes_[node] / 1000000.0; }

    int descriptor_ = -1;
    std::size_t size_ = 0;
    std::uint8_t* data_ = nullptr;
    std::uint32_t nodes_ = 0;
    std::uint32_t edges_ = 0;
    std::uint32_t scale_ = 0;
    std::uint32_t spainNodes_ = 0;
    std::uint32_t franceNodes_ = 0;
    std::uint32_t switzerlandNodes_ = 0;
    const std::int32_t* latitudes_ = nullptr;
    const std::int32_t* longitudes_ = nullptr;
    const std::uint32_t* offsets_ = nullptr;
    const std::uint32_t* targets_ = nullptr;
    const std::uint32_t* distances_ = nullptr;
    const std::uint32_t* durations_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: test_routing_pack GRAPH.owgr\n";
        return 2;
    }
    try {
        Graph graph(argv[1]);
        // Current Mallorca area -> Geneva. This covers the Palma/Barcelona ferry,
        // the Spain/France border and the France/Switzerland border in one test.
        graph.testRoute(39.56308, 2.65559, 46.20440, 6.14320);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Routing test failed: " << error.what() << '\n';
        return 1;
    }
}
