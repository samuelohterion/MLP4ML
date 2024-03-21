#ifndef MLP4ML_HPP
#define MLP4ML_HPP

#include "../AlgebraWithSTL/algebra.hpp"
// for matlab copy algebra.* to this directory an include it from here
// #include "algebra.hpp"
using namespace alg;

class MLP4ML {

    public:

        D
        eta,
        weightsMin,
        weightsMax;

        VD
        input;

        MD
        outputs,
        netsums,
        deltas,
        bias;

        TD
        weights;

        MLP4ML
        & softmax(VD const & pNetSums, VD & pOutputs) {
            D  m = *std::max_element(pNetSums.cbegin(), pNetSums.cend());
            std::transform(pNetSums.cbegin(), pNetSums.cend(), pOutputs.begin(), [m](D const &x){return exp(x - m);});
            
            D  s = alg::sum(pOutputs);
            std::transform(pOutputs.cbegin(), pOutputs.cend(), pOutputs.begin(), [s](D const &x){return x / s;});

            return *this;
        };
        
    public:

        MLP4ML(IDX const & pLayerSizes, D const &pEta = .001, D const &pWeightsMin = -.1, D const &pWeightsMax = +.1) :
        eta(pEta),
        weightsMin(pWeightsMin),
        weightsMax(pWeightsMax),
        input(pLayerSizes[0], 0.) {
            for (SIZE layerID = 1; layerID < pLayerSizes.size(); ++ layerID) {
                outputs.push_back(vcnst(pLayerSizes[layerID], 0.));
                netsums.push_back(vcnst(pLayerSizes[layerID], 0.));
                deltas.push_back(vcnst(pLayerSizes[layerID], 0.));
                bias.push_back(weightsMin + (weightsMax - weightsMin) * vrnd(pLayerSizes[layerID]));
                weights.push_back(weightsMin + (weightsMax - weightsMin) * mrnd(pLayerSizes[layerID], pLayerSizes[layerID-1]));
            }
        }

        MLP4ML
        & remember(VD const & pPattern) {

            input.assign(pPattern.cbegin(), pPattern.cend());
            
            SIZE
            layerID = 0;
            
            D
            s;
            
            for (SIZE toID = 0; toID < len(outputs[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < len(input); ++fromID) {
                    s += weights[layerID][toID][fromID] * input[fromID];
                }
                netsums[layerID][toID] = s - bias[layerID][toID];
                outputs[layerID][toID] = 0. < netsums[layerID][toID] ? netsums[layerID][toID] : 0.;
            }
            
            while (++ layerID < len(outputs) - 1) {
                for (SIZE toID = 0; toID < len(outputs[layerID]); ++toID) {
                    s = 0;
                    for (SIZE fromID = 0; fromID < len(outputs[layerID - 1]); ++fromID) {
                        s += weights[layerID][toID][fromID] * outputs[layerID - 1][fromID];
                    }
                    netsums[layerID][toID] = s - bias[layerID][toID];
                    outputs[layerID][toID] = 0. < netsums[layerID][toID] ? netsums[layerID][toID] : 0.;
                }
            }

            for (SIZE toID = 0; toID < len(outputs[layerID]); ++toID) {
                s = 0;
                for (SIZE fromID = 0; fromID < len(outputs[layerID - 1]); ++fromID) {
                    s += weights[layerID][toID][fromID] * outputs[layerID - 1][fromID];
                }
                netsums[layerID][toID] = s - bias[layerID][toID];
            }
            
            return softmax(netsums[layerID], outputs[layerID]);
        }

        MLP4ML
        & teach(VD const & pTeacher) {

            SIZE
            layerID = outputs.size() - 1;

            for (SIZE neuronID = 0; neuronID < deltas[layerID].size(); ++ neuronID) {
                deltas[layerID][neuronID] = pTeacher[neuronID] - outputs[layerID][neuronID];
            }

            D
            s;

            while (0 < layerID) {
                 -- layerID;
                for (SIZE neuronFromID = 0; neuronFromID < deltas[layerID].size(); ++ neuronFromID) {
                    s = 0.;
                    for (SIZE neuronToID = 0; neuronToID < deltas[layerID + 1].size(); ++ neuronToID) {
                        s += deltas[layerID + 1][neuronToID] * weights[layerID + 1][neuronToID][neuronFromID];
                    }
                    deltas[layerID][neuronFromID] =  (0 < outputs[layerID][neuronFromID] ? 1 : .001) * s;
                }
            }

            for (SIZE toID = 0; toID < len(outputs[layerID]); ++ toID) {
                for (SIZE fromID = 0; fromID < len(input); ++ fromID) {
                    weights[layerID][toID][fromID] += eta * input[fromID] * deltas[layerID][toID];
                }
                bias[layerID][toID] -= eta * deltas[layerID][toID]; 
            }
            
            while (++ layerID < len(outputs)) {
                for (SIZE toID = 0; toID < len(outputs[layerID]); ++ toID) {
                    for (SIZE fromID = 0; fromID < len(outputs[layerID - 1]); ++ fromID) {
                        weights[layerID][toID][fromID] += eta * outputs[layerID-1][fromID] * deltas[layerID][toID];
                    }
                    bias[layerID][toID] -= eta * deltas[layerID][toID]; 
                }
            }

            return *this;
        }
};

#endif //MLP4ML