#include "mlpclassifier4matlab.hpp"

// void
// pr(MLP4ML & pMLP, MD const & pPatterns, MD const & pTeachers) {
//     for (SIZE i = 0; i < len(pPatterns); ++ i) {
//         pMLP.remember(pPatterns[i]);
//         std::cout << std::setw(3) << std::noshowpos << std::left;
//         for (SIZE j = 0; j < len(pPatterns[i]); ++ j) {
//             std::cout << std::setw(3) << std::showpos << pPatterns[i][j];
//         }
//         std::cout << std::setw(3) << std::noshowpos;
//         std::cout << "[  " << pTeachers[i] << "] " << std::left;
//         for (SIZE outputID = 0; outputID < len(pTeachers); ++ outputID) {

//             std::cout << std::setw(6) << round(pMLP.output()[outputID], 2);
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl << std::noshowpos << std::setw(0);
// }

void
pr(MLPClassifier4MatLab & pMLP, MD const & pPatterns, Vec<SIZE> const & pLabels) {
    for (SIZE i = 0; i < len(pPatterns); ++ i) {
        pMLP.remember(pPatterns[i]);
        std::cout << std::setw(3) << std::noshowpos << std::left;
        for (SIZE j = 0; j < len(pPatterns[i]); ++ j) {
            std::cout << std::setw(3) << std::showpos << pPatterns[i][j];
        }
        std::cout << std::setw(3) << std::noshowpos << std::right << "[" << std::setw(2) << pLabels[i] << "] " << std::setw(2) << pMLP.label() << "  ";
        std::cout << "   " << std::right << std::setw(5) << round(100 * pMLP.output()[pMLP.label()], 2) << "%" << "  ";

        for (SIZE outputID = 0; outputID < len(pMLP.output()); ++ outputID) {

            std::cout << std::setw(6) << round(100 * pMLP.output()[outputID], 1);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::noshowpos << std::setw(0);
}

struct
SERIALIZED_MATRIX {
    VD vec;
    Vec<SIZE> sizes;
};

SERIALIZED_MATRIX
serialize(MD const & pMatrix) {

    SERIALIZED_MATRIX
    sm {VD(),Vec<SIZE>()};

    for (SIZE i = 0; i < len(pMatrix); ++ i) {
        for (SIZE j = 0; j < len(pMatrix[i]); ++ j) {
            sm.vec.push_back(pMatrix[i][j]);
        }
        sm.sizes.push_back(len(pMatrix[i]));
    }

    return sm;
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

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Vec<SIZE>
    labels(len(patterns));

    for (SIZE i=0;i<len(labels);++i){labels[i]=i;}

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    SIZE const 
    loopEnd   = 10000,
    loopPrint = 1000;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << "ORIGINAL INPUT" << std::endl;

    srand(3);

    MLPClassifier4MatLab
    mlp4ml({4, 4, 16});

    SIZE
    loop = 0;
    
    for (; loop < loopEnd; ++ loop) {

        SIZE
        id = rand() & 0x0f;

        mlp4ml.teach(patterns[id], labels[id]);

        if (!(loop % loopPrint)) {
            std::cout << "loop: " << loop << std::endl;
            pr(mlp4ml, patterns, labels);
        }
    }

    std::cout << "loop: " << loop << std::endl;
    pr(mlp4ml, patterns, labels);
    std::cout << std::endl;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    std::cout << "STANDARDIZED INPUT" << std::endl;

    srand(3);

    MLPClassifier4MatLab
    mlp4mlStd({4, 4, 16});

    loop = 0;
    
    for (; loop < loopEnd; ++ loop) {

        SIZE
        id = rand() & 0x0f;

        mlp4mlStd.teach(stdPatterns[id], labels[id]);

        if (!(loop % loopPrint)) {
            std::cout << "loop: " << loop << std::endl;
            pr(mlp4mlStd, stdPatterns, labels);
        }
    }

    std::cout << "loop: " << loop << std::endl;
    pr(mlp4mlStd, stdPatterns, labels);
    std::cout << std::endl;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << "STANDARDIZED INPUT - BATCH" << std::endl;

    SERIALIZED_MATRIX
    sm = serialize(stdPatterns);

    srand(3);

    MLPClassifier4MatLab
    mlp4mlStd2({4, 4, 16}, .001, -.001, +.001);

    loop = 0;
    
    for (; loop < loopEnd; ++ loop) {

        SIZE
        id = rand() & 0x0f;

        mlp4mlStd2.teach(stdPatterns[id], labels[id]);

        if (!(loop % loopPrint)) {
            std::cout << "loop: " << loop << std::endl;
            pr(mlp4mlStd2, stdPatterns, labels);
        }
    }

    for (loop = 0; loop < loopEnd; ++ loop) {

        mlp4mlStd2.teachBatch(sm.vec, labels);

        if (!(loop % loopPrint)) {
            std::cout << "loop: " << loop << std::endl;
            pr(mlp4mlStd2, stdPatterns, labels);
        }
    }

    std::cout << "loop: " << loop << std::endl;
    pr(mlp4mlStd2, stdPatterns, labels);
    std::cout << std::endl;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // std::cout << "STANDARDIZED INPUT - MINI-BATCHES" << std::endl;

    // srand(3);

    // MLP4ML
    // mlp4mlStd3({4, 4, 16});

    // loop = 0;
    
    // for (; loop < loopEnd; ++ loop) {

    //     SIZE
    //     id = rand() & 0x0f;

    //     mlp4mlStd2.teach(stdPatterns[id], labels[id]);

    //     if (!(loop % loopPrint)) {
    //         std::cout << "loop: " << loop << std::endl;
    //         pr(mlp4mlStd3, stdPatterns, labels);
    //     }
    // }

    // for (loop = 0; loop < loopEnd; ++ loop) {

    //     mlp4mlStd2.teachBatch(sm.vec, labels);

    //     if (!(loop % loopPrint)) {
    //         std::cout << "loop: " << loop << std::endl;
    //         pr(mlp4mlStd3, stdPatterns, labels);
    //     }
    // }

    // std::cout << "loop: " << loop << std::endl;
    // pr(mlp4mlStd3, stdPatterns, labels);
    // std::cout << std::endl;

    return 0;
}