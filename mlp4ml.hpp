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

        D const
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
        & resetAccumulators() {
            for (SIZE layerID = 0; layerID < len(this->__sumWeights); ++ layerID) {
                for (SIZE toID = 0; toID < len(this->__sumWeights[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < len(this->__sumWeights[layerID][toID]); ++ fromID) {
                        this->__sumWeights[layerID][toID][fromID] = 0.;
                    }
                    this->__sumBias[layerID][toID] = 0.;
                }
            }

            return *this;
        }

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
            for (SIZE layerID = 1; layerID < len(pLayerSizes); ++ layerID) {
                this->__out.push_back(vcnst(pLayerSizes[layerID], 0.));
                this->__net.push_back(vcnst(pLayerSizes[layerID], 0.));
                this->__delta.push_back(vcnst(pLayerSizes[layerID], 0.));
                this->__bias.push_back(weightsMin + (weightsMax - weightsMin) * vrnd(pLayerSizes[layerID]));
                this->__sumBias.push_back(vcnst(pLayerSizes[layerID]));
                this->__weights.push_back(weightsMin + (weightsMax - weightsMin) * mrnd(pLayerSizes[layerID], pLayerSizes[layerID-1]));
                this->__sumWeights.push_back(mcnst(pLayerSizes[layerID], pLayerSizes[layerID-1]));
            }
        }

        VD const
        & output() const {
            return this->__out[len(this->__out) - 1];
        }
        
        MLP4ML
        & remember(VD const & pPattern) {

            return this->remember(pPattern.data());
        }

        MLP4ML
        & remember(D const * const & pPatternData) {

            this->__input = pPatternData;
            
            SIZE
            layerID = 0;
            
            D
            s;
            
            for (SIZE toID = 0; toID < len(this->__out[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < this->__numOfInputs; ++fromID) {
                    s += this->__weights[layerID][toID][fromID] * this->__input[fromID];
                }
                this->__net[layerID][toID] = s - this->__bias[layerID][toID];
                this->__out[layerID][toID] = 0. < this->__net[layerID][toID] ? this->__net[layerID][toID] : 0.;
            }
            
            while (++ layerID < len(this->__out) - 1) {
                for (SIZE toID = 0; toID < len(this->__out[layerID]); ++toID) {
                    s = 0;
                    for (SIZE fromID = 0; fromID < len(this->__out[layerID - 1]); ++fromID) {
                        s += this->__weights[layerID][toID][fromID] * this->__out[layerID - 1][fromID];
                    }
                    this->__net[layerID][toID] = s - this->__bias[layerID][toID];
                    this->__out[layerID][toID] = 0. < this->__net[layerID][toID] ? this->__net[layerID][toID] : 0.;
                }
            }

            for (SIZE toID = 0; toID < len(this->__out[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < len(this->__out[layerID - 1]); ++fromID) {
                    s += this->__weights[layerID][toID][fromID] * this->__out[layerID - 1][fromID];
                }
                this->__net[layerID][toID] = s - this->__bias[layerID][toID];
            }
            
            return softmax(this->__net[layerID], this->__out[layerID]);
        }

        MLP4ML
        & teachBatch(VD const & pPatterns, Vec<SIZE> const & pLabels) {

            resetAccumulators();

            D const
            * patternsPtr;
            
            for (SIZE batchID = 0; batchID < len(pLabels); ++ batchID) {

                // remember patterns
                SIZE
                layerID = 0;

                patternsPtr = pPatterns.data() + batchID * this->__numOfInputs;
                
                remember(patternsPtr);

                D
                s;
                
                // for (SIZE toID = 0; toID < len(this->__out[layerID]); ++toID) {
                //     s = 0;
                //     for (SIZE fromID = 0; fromID < this->__numOfInputs; ++fromID) {
                //         s += this->__weights[layerID][toID][fromID] * patternsPtr[fromID];
                //     }
                //     this->__net[layerID][toID] = s - this->__bias[layerID][toID];
                //     this->__out[layerID][toID] = 0. < this->__net[layerID][toID] ? this->__net[layerID][toID] : 0.;
                // }
                
                // while (++ layerID < len(this->__out) - 1) {
                //     for (SIZE toID = 0; toID < len(this->__out[layerID]); ++toID) {
                //         s = 0;
                //         for (SIZE fromID = 0; fromID < len(this->__out[layerID - 1]); ++fromID) {
                //             s += this->__weights[layerID][toID][fromID] * this->__out[layerID - 1][fromID];
                //         }
                //         this->__net[layerID][toID] = s - this->__bias[layerID][toID];
                //         this->__out[layerID][toID] = 0. < this->__net[layerID][toID] ? this->__net[layerID][toID] : 0.;
                //     }
                // }

                // for (SIZE toID = 0; toID < len(this->__out[layerID]); ++toID) {
                //     s = 0;
                //     for (SIZE fromID = 0; fromID < len(this->__out[layerID - 1]); ++fromID) {
                //         s += this->__weights[layerID][toID][fromID] * this->__out[layerID - 1][fromID];
                //     }
                //     this->__net[layerID][toID] = s - this->__bias[layerID][toID];
                // }
                
                // softmax(this->__net[layerID], this->__out[layerID]);

                // teach                
                layerID = len(this->__out) - 1;

                for (SIZE neuronID = 0; neuronID < len(this->__delta[layerID]); ++ neuronID) {
                    this->__delta[layerID][neuronID] = -this->__out[layerID][neuronID];                
                }
                this->__delta[layerID][pLabels[batchID]] += 1.;

                while (0 < layerID) {
                    -- layerID;
                    for (SIZE neuronFromID = 0; neuronFromID < this->__delta[layerID].size(); ++ neuronFromID) {
                        s = 0.;
                        for (SIZE neuronToID = 0; neuronToID < this->__delta[layerID + 1].size(); ++ neuronToID) {
                            s += this->__delta[layerID + 1][neuronToID] * this->__weights[layerID + 1][neuronToID][neuronFromID];
                        }
                        this->__delta[layerID][neuronFromID] = (0 < this->__out[layerID][neuronFromID] ? 1 : .001) * s;
                    }
                }

                for (SIZE toID = 0; toID < len(this->__out[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < this->__numOfInputs; ++ fromID) {
                        this->__sumWeights[layerID][toID][fromID] += patternsPtr[fromID] * this->__delta[layerID][toID];
                    }
                    this->__sumBias[layerID][toID] -= this->__delta[layerID][toID]; 
                }
                
                while (++ layerID < len(this->__out)) {
                    for (SIZE toID = 0; toID < len(this->__out[layerID]); ++ toID) {
                        for (SIZE fromID = 0; fromID < len(this->__out[layerID - 1]); ++ fromID) {
                            this->__sumWeights[layerID][toID][fromID] += this->__out[layerID-1][fromID] * this->__delta[layerID][toID];
                        }
                        this->__sumBias[layerID][toID] -= this->__delta[layerID][toID]; 
                    }
                }
    
                D
                factor = eta / static_cast<D>(len(pLabels));
                    
                while (0 < layerID) {
                    -- layerID;
                    for (SIZE toID = 0; toID < len(this->__weights[layerID]); ++ toID) {
                        for (SIZE fromID = 0; fromID < len(this->__weights[layerID][toID]); ++ fromID) {
                            this->__weights[layerID][toID][fromID] += factor * this->__sumWeights[layerID][toID][fromID];
                        }
                        this->__bias[layerID][toID] -= factor * this->__sumBias[layerID][toID]; 
                    }
                }
            }

            return *this;
        }

        MLP4ML
        & teach(SIZE const & pLabel) {

            SIZE
            layerID = len(this->__out) - 1;

            for (SIZE neuronID = 0; neuronID < this->__delta[layerID].size(); ++ neuronID) {
                this->__delta[layerID][neuronID] = -this->__out[layerID][neuronID];                
            }
            this->__delta[layerID][pLabel] += 1.;

            D
            s;

            while (0 < layerID) {
                 -- layerID;
                for (SIZE neuronFromID = 0; neuronFromID < this->__delta[layerID].size(); ++ neuronFromID) {
                    s = 0.;
                    for (SIZE neuronToID = 0; neuronToID < this->__delta[layerID + 1].size(); ++ neuronToID) {
                        s += this->__delta[layerID + 1][neuronToID] * this->__weights[layerID + 1][neuronToID][neuronFromID];
                    }
                    this->__delta[layerID][neuronFromID] = (0 < this->__out[layerID][neuronFromID] ? 1 : .001) * s;
                }
            }

            for (SIZE toID = 0; toID < len(this->__out[layerID]); ++ toID) {
                for (SIZE fromID = 0; fromID < this->__numOfInputs; ++ fromID) {
                    this->__weights[layerID][toID][fromID] += eta * this->__input[fromID] * this->__delta[layerID][toID];
                }
                this->__bias[layerID][toID] -= eta * this->__delta[layerID][toID]; 
            }
            
            while (++ layerID < len(this->__out)) {
                for (SIZE toID = 0; toID < len(this->__out[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < len(this->__out[layerID - 1]); ++ fromID) {
                        this->__weights[layerID][toID][fromID] += eta * this->__out[layerID-1][fromID] * this->__delta[layerID][toID];
                    }
                    this->__bias[layerID][toID] -= eta * this->__delta[layerID][toID]; 
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
            & o = this->__out[len(this->__out) - 1];

            return static_cast<SIZE>(std::max_element(o.cbegin(), o.cend()) - o.cbegin());
        }
};

#endif //MLP4ML