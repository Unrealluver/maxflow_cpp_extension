#include "graph.h"

int main() {

    Graph<float, float, float> g = Graph<float, float, float>(41 * 41 + 2, 41 * 41 * (11 + 2) * 2);
    g.maxflow_init();
//    g.maxflow();
    return 1;

}