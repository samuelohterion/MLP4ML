#include "mlp4ml.hpp"

void
pr(MLP4ML & pMLP, MD const & pPatterns, MD const & pTeachers) {
    for (SIZE i = 0; i < len(pPatterns); ++ i) {
        pMLP.remember(pPatterns[i]);
        std::cout << std::setw(3) << std::noshowpos << std::left;
        for (SIZE j = 0; j < len(pPatterns[i]); ++ j) {
            std::cout << std::setw(3) << std::showpos << pPatterns[i][j];
        }
        std::cout << std::setw(3) << std::noshowpos;
        std::cout << "[  " << pTeachers[i] << "] " << std::left;
        for (SIZE outputID = 0; outputID < len(pTeachers); ++ outputID) {

            std::cout << std::setw(6) << round(pMLP.outputs[pMLP.outputs.size() - 1][outputID], 2);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::noshowpos << std::setw(0);
}

int
main() {

    MD
    patterns = {
        {0,0,0,0},
        {0,0,0,1},
        {0,0,1,0},
        {0,0,1,1},
        {0,1,0,0},
        {0,1,0,1},
        {0,1,1,0},
        {0,1,1,1},
        {1,0,0,0},
        {1,0,0,1},
        {1,0,1,0},
        {1,0,1,1},
        {1,1,0,0},
        {1,1,0,1},
        {1,1,1,0},
        {1,1,1,1}
    },
    teachers = eye(16),
    stdPatterns(4, VD(16, 0.));

    patterns = ~patterns;

    for (SIZE patternID = 0; patternID < len(patterns); ++ patternID) {
        D
        n = static_cast<D>(len(patterns[patternID])),
        mu = sum(patterns[patternID]) / n,
        sigma = sqrt((patterns[patternID] | patterns[patternID]) / n - mu * mu);

        stdPatterns[patternID] = (patterns[patternID] - mu) / sigma;    
    }

    stdPatterns = ~stdPatterns;
    patterns    = ~patterns;

    
    std::cout << "ORIGINAL INPUT" << std::endl;

    srand(3);

    MLP4ML
    mlp4ml({4, 4, 16}, .1, -.1, +.1);

    SIZE
    loop = 0;
    
    for (; loop < 10000; ++ loop) {

        SIZE
        id = rand() & 0x0f;

        mlp4ml.remember(patterns[id]);
        mlp4ml.teach(teachers[id]);

        if (!(loop % 1000)) {
            std::cout << "loop: " << loop << std::endl;
            pr(mlp4ml, patterns, teachers);
        }
    }

    std::cout << "loop: " << loop << std::endl;
    pr(mlp4ml, patterns, teachers);
    std::cout << std::endl;

    std::cout << "STANDARDIZED INPUT" << std::endl;

    srand(3);

    MLP4ML
    mlp4mlStd({4, 4, 16}, .1, -.1, +.1);

    loop = 0;
    
    for (; loop < 10000; ++ loop) {

        SIZE
        id = rand() & 0x0f;

        mlp4ml.remember(stdPatterns[id]);
        mlp4ml.teach(teachers[id]);

        if (!(loop % 1000)) {
            std::cout << "loop: " << loop << std::endl;
            pr(mlp4ml, stdPatterns, teachers);
        }
    }

    std::cout << "loop: " << loop << std::endl;
    pr(mlp4ml, stdPatterns, teachers);
    std::cout << std::endl;

    return 0;
}