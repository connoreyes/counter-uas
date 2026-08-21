#include <cstdio>
#include "stat.hpp"

int main() {
    // Case 1: 100 samples, values 1..100
    Stat a;
    for (int i = 1; i <= 100; ++i) {
        a.add(i);
    }
    printf("100 samples: p50=%.1f p95=%.1f p99=%.1f\n",
           a.percentile(0.50), a.percentile(0.95), a.percentile(0.99));

    // Case 2: 600 samples — ring only keeps the last 512
    Stat b;
    for (int i = 1; i <= 600; ++i) {
        b.add(i);
    }
    printf("600 samples: p50=%.1f p99=%.1f\n",
           b.percentile(0.50), b.percentile(0.99));

    // Case 3: empty
    Stat c;
    printf("empty: p50=%.1f\n", c.percentile(0.50));

    return 0;
}