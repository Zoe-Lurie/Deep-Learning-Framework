#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "src/tensor.h"

std::vector<std::pair<int, Tensor>>  readInput(std::string filename, size_t size){
    std::ifstream file(filename);

    std::vector<std::pair<int, Tensor>> data;

    std::string line;
    while(getline(file, line)){
        std::stringstream ls(line);

        std::string slabel;
        ls >> slabel;
        int label = std::stoi(slabel);

        std::vector<double> d(size, 0);
        std::string tmp;
        while(ls >> tmp){
            size_t index = std::stoi(tmp.substr(0, tmp.find(":")));
            int num = std::stoi(tmp.substr(tmp.find(":") + 1));
            d[index] = num;
        }

        data.push_back(std::make_pair(label ,Tensor({size}, d, false)));
    }

    return data;
}

std::pair<int, std::vector<Tensor>> predict(std::vector<Tensor> weights, Tensor t, int num_classes, int label){
    std::vector<Tensor> class_outputs;
    for(Tensor& w : weights){
        class_outputs.push_back(t.elementwiseMult(w).reduceSum());
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

    return std::make_pair(index, class_outputs);
}

std::vector<Tensor> train(std::vector<Tensor> weights, std::vector<std::pair<int, Tensor>> data, std::vector<std::pair<int, Tensor>> test_data, double req_acc, int num_classes, double learning_rate, int max_epochs, size_t num_features){
    double acc = 0;
    int epoch = 0;

    while(acc < req_acc && epoch < max_epochs){
        size_t m = 0;
        for(auto& d : data){
            auto p = predict(weights, d.second, num_classes, d.first);
            int index = p.first;
            std::vector<Tensor> outputs = p.second;
            
            if(index != d.first){
                for(int i = 0; i < num_classes; ++i){
                    outputs[i].backward();
                    weights[i] = Tensor({num_features}, weights[i].getData(), true) -
                        weights[i].getGradient() * learning_rate * Tensor({1}, outputs[i].getData()) / data.size();
                }
            }

            if(m % 1000 == 0)
                std::cout << "Training sample " << m << " processed\n";
            m ++;
        }
        
        size_t num_correct = 0;
        for(auto& t : test_data){
            auto p = predict(weights, t.second, num_classes, num_features);
            int index = p.first;
            if(index == t.first)
                num_correct ++;
        }
        acc = (double) num_correct / test_data.size();

        std::cout << "Epoch " << epoch << " complete with accuracy " << acc * 100 << "%\n";
        epoch ++;
    }

    return weights;
}

int main(){
    Tensor::setOmpNumThreads(8);

    size_t num_features = 784;
    int num_classes = 10;

    std::string data_file = "data/mnist";
    std::string test_data_file = "data/mnist.t";

    auto data = readInput(data_file, num_features);
    auto test_data = readInput(test_data_file, num_features);

    std::vector<Tensor> weights;
    for(int i = 0; i < num_classes; ++i)
        weights.push_back(Tensor::fillRandom({num_features}, 0, 0.1, true));

    auto final_weights = train(weights, data, test_data, 0.6, num_classes, 0.2, 100, num_features);

    return 0;
}

