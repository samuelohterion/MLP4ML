#ifndef MLP4ML_HPP
#define MLP4ML_HPP

#include "../AlgebraWithSTL/algebra.hpp"
#include<bits/stdc++.h>
// for matlab copy algebra.* to this directory an include it from here
// #include "algebra.hpp"
using namespace alg;

class MLP4ML {

    public:

        D
        eta,
        weightsMin,
        weightsMax;

    private:

        VD const
        * __input;

        SIZE
        __numOfInputs;

        MD
        __out,
        __net,
        __delta,
        __bias,
        __sumBias;

        TD
        __weights,
        __sumWeights;

        MLP4ML
        & softmax(VD const & pNetSums, VD & pOutputs) {

            D  m = *std::max_element(pNetSums.cbegin(), pNetSums.cend());
            std::transform(pNetSums.cbegin(), pNetSums.cend(), pOutputs.begin(), [m](D const &x){return exp(x - m);});
            
            D  s = alg::sum(pOutputs);
            std::transform(pOutputs.cbegin(), pOutputs.cend(), pOutputs.begin(), [s](D const &x){return x / s;});

            return *this;
        };
        
    public:

        MLP4ML(IDX const & pLayerSizes, D const &pEta = .1, D const &pWeightsMin = -.1, D const &pWeightsMax = +.1) :
        eta(pEta),
        weightsMin(pWeightsMin),
        weightsMax(pWeightsMax),
        __input(nullptr),
        __numOfInputs(pLayerSizes[0]) {
            for (SIZE layerID = 1; layerID < pLayerSizes.size(); ++ layerID) {
                __out.push_back(vcnst(pLayerSizes[layerID], 0.));
                __net.push_back(vcnst(pLayerSizes[layerID], 0.));
                __delta.push_back(vcnst(pLayerSizes[layerID], 0.));
                __bias.push_back(weightsMin + (weightsMax - weightsMin) * vrnd(pLayerSizes[layerID]));
                __sumBias.push_back(vcnst(pLayerSizes[layerID]));
                __weights.push_back(weightsMin + (weightsMax - weightsMin) * mrnd(pLayerSizes[layerID], pLayerSizes[layerID-1]));
                __sumWeights.push_back(mcnst(pLayerSizes[layerID], pLayerSizes[layerID-1]));
            }
        }

        VD const
        & output() const {
            return __out[len(__out) - 1];
        }
        
        MLP4ML
        & remember(VD const & pPattern) {

            // __input.assign(pPattern.cbegin(), pPattern.cend());
            __input = &pPattern;
            
            SIZE
            layerID = 0;
            
            D
            s;
            
            for (SIZE toID = 0; toID < len(__out[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < __input->size(); ++fromID) {
                    s += __weights[layerID][toID][fromID] * (*__input)[fromID];
                }
                __net[layerID][toID] = s - __bias[layerID][toID];
                __out[layerID][toID] = 0. < __net[layerID][toID] ? __net[layerID][toID] : 0.;
            }
            
            while (++ layerID < len(__out) - 1) {
                for (SIZE toID = 0; toID < len(__out[layerID]); ++toID) {
                    s = 0;
                    for (SIZE fromID = 0; fromID < len(__out[layerID - 1]); ++fromID) {
                        s += __weights[layerID][toID][fromID] * __out[layerID - 1][fromID];
                    }
                    __net[layerID][toID] = s - __bias[layerID][toID];
                    __out[layerID][toID] = 0. < __net[layerID][toID] ? __net[layerID][toID] : 0.;
                }
            }

            for (SIZE toID = 0; toID < len(__out[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < len(__out[layerID - 1]); ++fromID) {
                    s += __weights[layerID][toID][fromID] * __out[layerID - 1][fromID];
                }
                __net[layerID][toID] = s - __bias[layerID][toID];
            }
            
            return softmax(__net[layerID], __out[layerID]);
        }

        MLP4ML
        & remember(D const * const & pPatternBegin) {

            // __input.assign(pPattern.cbegin(), pPattern.cend());
            //__input = pPatternBegin;
            
            SIZE
            layerID = 0;
            
            D
            s;
            
            for (SIZE toID = 0; toID < len(__out[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < __numOfInputs; ++fromID) {
                    s += __weights[layerID][toID][fromID] * pPatternBegin[fromID];
                }
                __net[layerID][toID] = s - __bias[layerID][toID];
                __out[layerID][toID] = 0. < __net[layerID][toID] ? __net[layerID][toID] : 0.;
            }
            
            while (++ layerID < len(__out) - 1) {
                for (SIZE toID = 0; toID < len(__out[layerID]); ++toID) {
                    s = 0;
                    for (SIZE fromID = 0; fromID < len(__out[layerID - 1]); ++fromID) {
                        s += __weights[layerID][toID][fromID] * __out[layerID - 1][fromID];
                    }
                    __net[layerID][toID] = s - __bias[layerID][toID];
                    __out[layerID][toID] = 0. < __net[layerID][toID] ? __net[layerID][toID] : 0.;
                }
            }

            for (SIZE toID = 0; toID < len(__out[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < len(__out[layerID - 1]); ++fromID) {
                    s += __weights[layerID][toID][fromID] * __out[layerID - 1][fromID];
                }
                __net[layerID][toID] = s - __bias[layerID][toID];
            }
            
            return softmax(__net[layerID], __out[layerID]);
        }

        MLP4ML
        & teachBatch(VD const & pPatterns, Vec<SIZE> const & pLabels) {

            for (SIZE layerID = 0; layerID < len(__sumWeights); ++ layerID) {
                for (SIZE toID = 0; toID < len(__sumWeights[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < len(__sumWeights[layerID][toID]); ++ fromID) {
                        __sumWeights[layerID][toID][fromID] = 0.;
                    }
                    __sumBias[layerID][toID] = 0.;
                }
            }

            D const
            * patternsPtr;
            
            for (SIZE batchID = 0; batchID < len(pLabels); ++ batchID) {

                // remember patterns
                SIZE
                layerID = 0;

                patternsPtr = pPatterns.data() + batchID * __numOfInputs;
                
                D
                s;
                
                for (SIZE toID = 0; toID < len(__out[layerID]); ++toID) {
                    s = 0;
                    for (SIZE fromID = 0; fromID < __numOfInputs; ++fromID) {
                        s += __weights[layerID][toID][fromID] * patternsPtr[fromID];
                    }
                    __net[layerID][toID] = s - __bias[layerID][toID];
                    __out[layerID][toID] = 0. < __net[layerID][toID] ? __net[layerID][toID] : 0.;
                }
                
                while (++ layerID < len(__out) - 1) {
                    for (SIZE toID = 0; toID < len(__out[layerID]); ++toID) {
                        s = 0;
                        for (SIZE fromID = 0; fromID < len(__out[layerID - 1]); ++fromID) {
                            s += __weights[layerID][toID][fromID] * __out[layerID - 1][fromID];
                        }
                        __net[layerID][toID] = s - __bias[layerID][toID];
                        __out[layerID][toID] = 0. < __net[layerID][toID] ? __net[layerID][toID] : 0.;
                    }
                }

                for (SIZE toID = 0; toID < len(__out[layerID]); ++toID) {
                    s = 0;
                    for (SIZE fromID = 0; fromID < len(__out[layerID - 1]); ++fromID) {
                        s += __weights[layerID][toID][fromID] * __out[layerID - 1][fromID];
                    }
                    __net[layerID][toID] = s - __bias[layerID][toID];
                }
                
                softmax(__net[layerID], __out[layerID]);

                // teach                
                layerID = len(__out) - 1;

                for (SIZE neuronID = 0; neuronID < len(__delta[layerID]); ++ neuronID) {
                    __delta[layerID][neuronID] = -__out[layerID][neuronID];                
                }
                __delta[layerID][pLabels[batchID]] += 1.;

                while (0 < layerID) {
                    -- layerID;
                    for (SIZE neuronFromID = 0; neuronFromID < __delta[layerID].size(); ++ neuronFromID) {
                        s = 0.;
                        for (SIZE neuronToID = 0; neuronToID < __delta[layerID + 1].size(); ++ neuronToID) {
                            s += __delta[layerID + 1][neuronToID] * __weights[layerID + 1][neuronToID][neuronFromID];
                        }
                        __delta[layerID][neuronFromID] = (0 < __out[layerID][neuronFromID] ? 1 : .001) * s;
                    }
                }

                for (SIZE toID = 0; toID < len(__out[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < __numOfInputs; ++ fromID) {
                        __sumWeights[layerID][toID][fromID] += patternsPtr[fromID] * __delta[layerID][toID];
                    }
                    __sumBias[layerID][toID] -= __delta[layerID][toID]; 
                }
                
                while (++ layerID < len(__out)) {
                    for (SIZE toID = 0; toID < len(__out[layerID]); ++ toID) {
                        for (SIZE fromID = 0; fromID < len(__out[layerID - 1]); ++ fromID) {
                            __sumWeights[layerID][toID][fromID] += __out[layerID-1][fromID] * __delta[layerID][toID];
                        }
                        __sumBias[layerID][toID] -= __delta[layerID][toID]; 
                    }
                }
    
                D
                factor = eta / static_cast<D>(len(pLabels));
                    
                while (0 < layerID) {
                    -- layerID;
                    for (SIZE toID = 0; toID < len(__weights[layerID]); ++ toID) {
                        for (SIZE fromID = 0; fromID < len(__weights[layerID][toID]); ++ fromID) {
                            __weights[layerID][toID][fromID] += factor * __sumWeights[layerID][toID][fromID];
                        }
                        __bias[layerID][toID] -= factor * __sumBias[layerID][toID]; 
                    }
                }
            }

            return *this;
        }

        // MLP4ML
        // & teach(VD const & pTeacher) {

        //     SIZE
        //     layerID = __out.size() - 1;

        //     for (SIZE neuronID = 0; neuronID < __delta[layerID].size(); ++ neuronID) {
        //         __delta[layerID][neuronID] = pTeacher[neuronID] - __out[layerID][neuronID];
        //     }

        //     D
        //     s;

        //     while (0 < layerID) {
        //          -- layerID;
        //         for (SIZE neuronFromID = 0; neuronFromID < __delta[layerID].size(); ++ neuronFromID) {
        //             s = 0.;
        //             for (SIZE neuronToID = 0; neuronToID < __delta[layerID + 1].size(); ++ neuronToID) {
        //                 s += __delta[layerID + 1][neuronToID] * __weights[layerID + 1][neuronToID][neuronFromID];
        //             }
        //             __delta[layerID][neuronFromID] = (0 < __out[layerID][neuronFromID] ? 1 : .001) * s;
        //         }
        //     }

        //     for (SIZE toID = 0; toID < len(__out[layerID]); ++ toID) {
        //         for (SIZE fromID = 0; fromID < __input->size(); ++ fromID) {
        //             __weights[layerID][toID][fromID] += eta * (*__input)[fromID] * __delta[layerID][toID];
        //         }
        //         __bias[layerID][toID] -= eta * __delta[layerID][toID]; 
        //     }
            
        //     while (++ layerID < len(__out)) {
        //         for (SIZE toID = 0; toID < len(__out[layerID]); ++ toID) {
        //             for (SIZE fromID = 0; fromID < len(__out[layerID - 1]); ++ fromID) {
        //                 __weights[layerID][toID][fromID] += eta * __out[layerID-1][fromID] * __delta[layerID][toID];
        //             }
        //             __bias[layerID][toID] -= eta * __delta[layerID][toID]; 
        //         }
        //     }

        //     return *this;
        // }

        // MLP4ML
        // & teach(VD const & pPattern, VD const & pTeacher) {
        
        //     return remember(pPattern).teach(pTeacher);
        // }

        MLP4ML
        & teach(SIZE const & pLabel) {

            SIZE
            layerID = __out.size() - 1;

            for (SIZE neuronID = 0; neuronID < __delta[layerID].size(); ++ neuronID) {
                __delta[layerID][neuronID] = -__out[layerID][neuronID];                
            }
            __delta[layerID][pLabel] += 1.;

            D
            s;

            while (0 < layerID) {
                 -- layerID;
                for (SIZE neuronFromID = 0; neuronFromID < __delta[layerID].size(); ++ neuronFromID) {
                    s = 0.;
                    for (SIZE neuronToID = 0; neuronToID < __delta[layerID + 1].size(); ++ neuronToID) {
                        s += __delta[layerID + 1][neuronToID] * __weights[layerID + 1][neuronToID][neuronFromID];
                    }
                    __delta[layerID][neuronFromID] = (0 < __out[layerID][neuronFromID] ? 1 : .001) * s;
                }
            }

            for (SIZE toID = 0; toID < len(__out[layerID]); ++ toID) {
                for (SIZE fromID = 0; fromID < __input->size(); ++ fromID) {
                    __weights[layerID][toID][fromID] += eta * (*__input)[fromID] * __delta[layerID][toID];
                }
                __bias[layerID][toID] -= eta * __delta[layerID][toID]; 
            }
            
            while (++ layerID < len(__out)) {
                for (SIZE toID = 0; toID < len(__out[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < len(__out[layerID - 1]); ++ fromID) {
                        __weights[layerID][toID][fromID] += eta * __out[layerID-1][fromID] * __delta[layerID][toID];
                    }
                    __bias[layerID][toID] -= eta * __delta[layerID][toID]; 
                }
            }

            return *this;
        }

        MLP4ML
        & teach(VD const & pPattern, SIZE const & pLabel) {
        
            return remember(pPattern).teach(pLabel);
        }

        MLP4ML
        & teach(D const * const & pPatternData, SIZE const & pLabel) {
        
            return remember(pPatternData).teach(pLabel);
        }

        SIZE
        label() const {

            VD const
            & o = __out[len(__out) - 1];

            return static_cast<SIZE>(std::max_element(o.cbegin(), o.cend()) - o.cbegin());
        }
};

#endif //MLP4ML