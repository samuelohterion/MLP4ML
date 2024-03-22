#ifndef MLPCLASSIFIER4MATLAB_HPP_HPP
#define MLPCLASSIFIER4MATLAB_HPP_HPP

#include "../AlgebraWithSTL/algebra.hpp"
#include<bits/stdc++.h>
// for matlab copy algebra.* to this directory an include it from here
// #include "algebra.hpp"

using namespace alg;

class MLPClassifier4MatLab {

    public:

        D
        eta,
        weightsMin,
        weightsMax;

    private:

        D const
        * input__;

        SIZE
        numOfInputs__;

        MD
        out__,
        net__,
        delta__,
        bias__,
        sumBias__;

        TD
        weights__,
        sumWeights__;

        MLPClassifier4MatLab
        & resetAccumulators() {
            for (SIZE layerID = 0; layerID < len(this->sumWeights__); ++ layerID) {
                for (SIZE toID = 0; toID < len(this->sumWeights__[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < len(this->sumWeights__[layerID][toID]); ++ fromID) {
                        this->sumWeights__[layerID][toID][fromID] = 0.;
                    }
                    this->sumBias__[layerID][toID] = 0.;
                }
            }

            return *this;
        }

        MLPClassifier4MatLab
        & softmax(VD const & pNetSums, VD & pOutputs) {

            D  m = *std::max_element(pNetSums.cbegin(), pNetSums.cend());
            std::transform(pNetSums.cbegin(), pNetSums.cend(), pOutputs.begin(), [m](D const &x){return exp(x - m);});
            
            D  s = alg::sum(pOutputs);
            std::transform(pOutputs.cbegin(), pOutputs.cend(), pOutputs.begin(), [s](D const &x){return x / s;});

            return *this;
        };
        
    public:

        MLPClassifier4MatLab(IDX const & pLayerSizes, D const &pEta = .1, D const &pWeightsMin = -.1, D const &pWeightsMax = +.1) :
        eta(pEta),
        weightsMin(pWeightsMin),
        weightsMax(pWeightsMax),
        input__(nullptr),
        numOfInputs__(pLayerSizes[0]) {
            for (SIZE layerID = 1; layerID < len(pLayerSizes); ++ layerID) {
                this->out__.push_back(vcnst(pLayerSizes[layerID], 0.));
                this->net__.push_back(vcnst(pLayerSizes[layerID], 0.));
                this->delta__.push_back(vcnst(pLayerSizes[layerID], 0.));
                this->bias__.push_back(weightsMin + (weightsMax - weightsMin) * vrnd(pLayerSizes[layerID]));
                this->sumBias__.push_back(vcnst(pLayerSizes[layerID]));
                this->weights__.push_back(weightsMin + (weightsMax - weightsMin) * mrnd(pLayerSizes[layerID], pLayerSizes[layerID-1]));
                this->sumWeights__.push_back(mcnst(pLayerSizes[layerID], pLayerSizes[layerID-1]));
            }
        }

        VD const
        & output() const {
            return this->out__[len(this->out__) - 1];
        }
        
        MLPClassifier4MatLab
        & remember(VD const & pPattern) {

            return this->remember(pPattern.data());
        }

        MLPClassifier4MatLab
        & remember(D const * const & pPatternData) {

            this->input__ = pPatternData;
            
            SIZE
            layerID = 0;
            
            D
            s;
            
            for (SIZE toID = 0; toID < len(this->out__[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < this->numOfInputs__; ++fromID) {
                    s += this->weights__[layerID][toID][fromID] * this->input__[fromID];
                }
                this->net__[layerID][toID] = s - this->bias__[layerID][toID];
                this->out__[layerID][toID] = 0. < this->net__[layerID][toID] ? this->net__[layerID][toID] : 0.;
            }
            
            while (++ layerID < len(this->out__) - 1) {
                for (SIZE toID = 0; toID < len(this->out__[layerID]); ++toID) {
                    s = 0;
                    for (SIZE fromID = 0; fromID < len(this->out__[layerID - 1]); ++fromID) {
                        s += this->weights__[layerID][toID][fromID] * this->out__[layerID - 1][fromID];
                    }
                    this->net__[layerID][toID] = s - this->bias__[layerID][toID];
                    this->out__[layerID][toID] = 0. < this->net__[layerID][toID] ? this->net__[layerID][toID] : 0.;
                }
            }

            for (SIZE toID = 0; toID < len(this->out__[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < len(this->out__[layerID - 1]); ++fromID) {
                    s += this->weights__[layerID][toID][fromID] * this->out__[layerID - 1][fromID];
                }
                this->net__[layerID][toID] = s - this->bias__[layerID][toID];
            }
            
            return softmax(this->net__[layerID], this->out__[layerID]);
        }

        MLPClassifier4MatLab
        & teachBatch(VD const & pPatterns, Vec<SIZE> const & pLabels) {

            resetAccumulators();

            D const
            * patternsPtr;
            
            for (SIZE batchID = 0; batchID < len(pLabels); ++ batchID) {

                // remember patterns
                SIZE
                layerID = 0;

                patternsPtr = pPatterns.data() + batchID * this->numOfInputs__;
                
                remember(patternsPtr);

                D
                s;
                
                // for (SIZE toID = 0; toID < len(this->out__[layerID]); ++toID) {
                //     s = 0;
                //     for (SIZE fromID = 0; fromID < this->numOfInputs__; ++fromID) {
                //         s += this->weights__[layerID][toID][fromID] * patternsPtr[fromID];
                //     }
                //     this->net__[layerID][toID] = s - this->bias__[layerID][toID];
                //     this->out__[layerID][toID] = 0. < this->net__[layerID][toID] ? this->net__[layerID][toID] : 0.;
                // }
                
                // while (++ layerID < len(this->out__) - 1) {
                //     for (SIZE toID = 0; toID < len(this->out__[layerID]); ++toID) {
                //         s = 0;
                //         for (SIZE fromID = 0; fromID < len(this->out__[layerID - 1]); ++fromID) {
                //             s += this->weights__[layerID][toID][fromID] * this->out__[layerID - 1][fromID];
                //         }
                //         this->net__[layerID][toID] = s - this->bias__[layerID][toID];
                //         this->out__[layerID][toID] = 0. < this->net__[layerID][toID] ? this->net__[layerID][toID] : 0.;
                //     }
                // }

                // for (SIZE toID = 0; toID < len(this->out__[layerID]); ++toID) {
                //     s = 0;
                //     for (SIZE fromID = 0; fromID < len(this->out__[layerID - 1]); ++fromID) {
                //         s += this->weights__[layerID][toID][fromID] * this->out__[layerID - 1][fromID];
                //     }
                //     this->net__[layerID][toID] = s - this->bias__[layerID][toID];
                // }
                
                // softmax(this->net__[layerID], this->out__[layerID]);

                // teach                
                layerID = len(this->out__) - 1;

                for (SIZE neuronID = 0; neuronID < len(this->delta__[layerID]); ++ neuronID) {
                    this->delta__[layerID][neuronID] = -this->out__[layerID][neuronID];                
                }
                this->delta__[layerID][pLabels[batchID]] += 1.;

                while (0 < layerID) {
                    -- layerID;
                    for (SIZE neuronFromID = 0; neuronFromID < this->delta__[layerID].size(); ++ neuronFromID) {
                        s = 0.;
                        for (SIZE neuronToID = 0; neuronToID < this->delta__[layerID + 1].size(); ++ neuronToID) {
                            s += this->delta__[layerID + 1][neuronToID] * this->weights__[layerID + 1][neuronToID][neuronFromID];
                        }
                        this->delta__[layerID][neuronFromID] = (0 < this->out__[layerID][neuronFromID] ? 1 : .001) * s;
                    }
                }

                for (SIZE toID = 0; toID < len(this->out__[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < this->numOfInputs__; ++ fromID) {
                        this->sumWeights__[layerID][toID][fromID] += patternsPtr[fromID] * this->delta__[layerID][toID];
                    }
                    this->sumBias__[layerID][toID] -= this->delta__[layerID][toID]; 
                }
                
                while (++ layerID < len(this->out__)) {
                    for (SIZE toID = 0; toID < len(this->out__[layerID]); ++ toID) {
                        for (SIZE fromID = 0; fromID < len(this->out__[layerID - 1]); ++ fromID) {
                            this->sumWeights__[layerID][toID][fromID] += this->out__[layerID-1][fromID] * this->delta__[layerID][toID];
                        }
                        this->sumBias__[layerID][toID] -= this->delta__[layerID][toID]; 
                    }
                }
    
                D
                factor = eta / static_cast<D>(len(pLabels));
                    
                while (0 < layerID) {
                    -- layerID;
                    for (SIZE toID = 0; toID < len(this->weights__[layerID]); ++ toID) {
                        for (SIZE fromID = 0; fromID < len(this->weights__[layerID][toID]); ++ fromID) {
                            this->weights__[layerID][toID][fromID] += factor * this->sumWeights__[layerID][toID][fromID];
                        }
                        this->bias__[layerID][toID] -= factor * this->sumBias__[layerID][toID]; 
                    }
                }
            }

            return *this;
        }

        MLPClassifier4MatLab
        & teach(SIZE const & pLabel) {

            SIZE
            layerID = len(this->out__) - 1;

            for (SIZE neuronID = 0; neuronID < this->delta__[layerID].size(); ++ neuronID) {
                this->delta__[layerID][neuronID] = -this->out__[layerID][neuronID];                
            }
            this->delta__[layerID][pLabel] += 1.;

            D
            s;

            while (0 < layerID) {
                 -- layerID;
                for (SIZE neuronFromID = 0; neuronFromID < this->delta__[layerID].size(); ++ neuronFromID) {
                    s = 0.;
                    for (SIZE neuronToID = 0; neuronToID < this->delta__[layerID + 1].size(); ++ neuronToID) {
                        s += this->delta__[layerID + 1][neuronToID] * this->weights__[layerID + 1][neuronToID][neuronFromID];
                    }
                    this->delta__[layerID][neuronFromID] = (0 < this->out__[layerID][neuronFromID] ? 1 : .001) * s;
                }
            }

            for (SIZE toID = 0; toID < len(this->out__[layerID]); ++ toID) {
                for (SIZE fromID = 0; fromID < this->numOfInputs__; ++ fromID) {
                    this->weights__[layerID][toID][fromID] += eta * this->input__[fromID] * this->delta__[layerID][toID];
                }
                this->bias__[layerID][toID] -= eta * this->delta__[layerID][toID]; 
            }
            
            while (++ layerID < len(this->out__)) {
                for (SIZE toID = 0; toID < len(this->out__[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < len(this->out__[layerID - 1]); ++ fromID) {
                        this->weights__[layerID][toID][fromID] += eta * this->out__[layerID-1][fromID] * this->delta__[layerID][toID];
                    }
                    this->bias__[layerID][toID] -= eta * this->delta__[layerID][toID]; 
                }
            }

            return *this;
        }

        MLPClassifier4MatLab
        & teach(VD const & pPattern, SIZE const & pLabel) {
        
            return remember(pPattern).teach(pLabel);
        }

        MLPClassifier4MatLab
        & teach(D const * const & pPatternData, SIZE const & pLabel) {
        
            return remember(pPatternData).teach(pLabel);
        }

        SIZE
        label() const {

            VD const
            & o = this->out__[len(this->out__) - 1];

            return static_cast<SIZE>(std::max_element(o.cbegin(), o.cend()) - o.cbegin());
        }
};

#endif //MLPCLASSIFIER4MATLAB_HPP