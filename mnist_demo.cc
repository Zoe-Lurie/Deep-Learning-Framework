#include <utility>
#include <vector>
#include <iostream>

#include "src/tensor.h"

std::vector<std::pair<int, Tensor>>  readInput(std::string filename){


}

std::pair<int, Tensor> predict(std::vector<Tensor> weights, Tensor t, int num_classes){
    std::vector<Tensor> class_outputs;
    for(Tensor& w : weights){
        class_outputs.push_back(t.matmul(w).reduceSum());
    }

    int index = 0;
    double max = class_outputs[0].getData()[0];

    for(int i = 1; i < num_classes; ++i){
        double output = class_outputs[i].getData()[0];
        if(output > max){
            index = i;
            max = output;
        }
    }

    return std::make_pair(index, class_outputs[index]);
}

void train(std::vector<Tensor> weights, std::vector<std::pair<int, Tensor>> data, std::vector<std::pair<int, Tensor>> test_data, double req_acc, int num_classes, double learning_rate, int max_epochs){
    double acc = 0;
    int epoch = 0;

    while(acc < req_acc && epoch < max_epochs){
        size_t m = 0;
        for(auto& d : data){
            auto p = predict(weights, d.second, num_classes);
            int index = p.first;
            Tensor output = p.second;
            
            if(index != d.first){
                output.backward();
                for(int i = 0; i < num_classes; ++i){
                    weights[i] = weights[i] - weights[i].getGradient() * learning_rate;
                }
            }

            std::cout << "Training sample " << m << " processed";
        }
        
        size_t num_correct = 0;
        for(auto& t : test_data){
            auto p = predict(weights, t.second, num_classes);
            int index = p.first;
            if(index == t.first)
                num_correct ++;
        }
        acc = (double) num_correct / test_data.size();

        std::cout << "Epoch " << epoch << " complete with accuracy " << acc;
        epoch ++;
    }
}

int main(){
    //Tensor::setOmpNumThreads(8);

    std::string data_file = "data/mnist.t";
    std::string test_data_file = "data/mnist.t";

    auto data = readInput(data_file);
    auto test_data = readInput(test_data_file);

    size_t num_features = 200;
    int num_classes = 10;
    std::vector<Tensor> weights;
    for(int i = 0; i < num_classes; ++i)
        weights.push_back(Tensor::fillRandom({num_features}, 0, 0.1, true));


    return 0;
}

