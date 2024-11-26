/**
 * @file tensor.h
 * @brief Defines the Tensor class and related structures for multidimensional array manipulation with support
 * for autodifferentiation, multithreaded CPU and GPU algorithms, and lazy evaluation.
 *
 * @note Documentation of methods in this file was generated with the help of ChatGPT with the following queries, the output of which were then modified:
 *      Generate documentation for the public methods of the following header file: **This file attached**
 *      can you save this as docstrings I can copy and paste back into my original file?
 *      you missed binarize, neg, pow, softmax, output the docstrings of just those
 *      you missed add, subtract, elementwisemult, elementwisedivision, output the docstrings of just those
 * 
 * @author Zoe Lurie
 * @date November 2024
 */

#ifndef TENSORH
#define TENSORH

#include <vector>
#include <memory>

struct TensorContents;

typedef std::vector<size_t> vDims;
typedef std::shared_ptr<double> vDataPtr;
typedef std::shared_ptr<TensorContents> TensorContentsPtr;

enum deviceOptions {CPU, GPU, DEFAULTDEVICE};

/**
 * @brief Represents a multidimensional array with support for gradient computations and device allocation.
 * 
 * Provides methods for tensor manipulation, mathematical operations, and gradient-based optimization.
 */
class Tensor{
    friend struct TensorContents;
    friend class TensorReshape;
    friend class TensorReduceSum;
    private:
        TensorContentsPtr contents;

        Tensor(TensorContentsPtr);
        vDataPtr eval();

    public:
        /**
         * @brief Constructs a tensor with specified dimensions, data, and options.
         * 
         * @param dimensions Shape of the tensor.
         * @param data Initial values for the tensor.
         * @param saveGradient Whether to compute and save gradients for this tensor (default: false).
         * @param device Device to allocate the tensor (CPU, GPU, or default: DEFAULTDEVICE).
         */
        Tensor(vDims, std::vector<double> data, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Prints the tensor's contents to the console.
         */
        void print();

        /**
         * @brief Retrieves the tensor's data as a 1D vector.
         * @return The tensor's data.
         */
        std::vector<double> getData();

        /**
         * @brief Returns the dimensions of the tensor.
         * @return The dimensions of the tensor.
         */
        vDims getDims();

        #ifdef OMP
            /**
             * @brief Sets the number of threads for omp globally
             * @param numThreads Number of threads to use.
             */
            static void setOmpNumThreads(int numThreads);
        #endif

        /**
         * @brief Creates a tensor filled with ones.
         * 
         * @param dimensions Shape of the tensor.
         * @param saveGradient Whether to compute gradients (default: false).
         * @param device Device to allocate the tensor (default: DEFAULTDEVICE).
         * @return A tensor filled with ones.
         */
        static Tensor ones(vDims, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Creates a tensor filled with ones.
         * 
         * @param dimensions Shape of the tensor.
         * @param saveGradient Whether to compute gradients (default: false).
         * @param device Device to allocate the tensor (default: DEFAULTDEVICE).
         * @return A tensor filled with ones.
         */
        static Tensor zeroes(vDims, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Creates a tensor filled with a specific value.
         * 
         * @param dimensions Shape of the tensor.
         * @param value Value to fill the tensor with.
         * @param saveGradient Whether to compute gradients (default: false).
         * @param device Device to allocate the tensor (default: DEFAULTDEVICE).
         * @return A tensor filled with the specified value.
         */
        static Tensor fill(vDims, double n, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Performs backpropagation to compute gradients.
         * 
         * @param grad Gradient to propagate (default: a tensor of scalar one).
         */
        void backward(Tensor grad = Tensor::ones({1}));

        /**
         * @brief Retrieves the gradient associated with the tensor.
         * @return The gradient tensor.
         */
        Tensor getGradient();

        /**
         * @brief Reshapes the tensor to new dimensions.
         * 
         * @param dimensions New shape for the tensor.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the reshaped tensor (default: DEFAULTDEVICE).
         * @return The reshaped tensor.
         */
        Tensor reshape(vDims, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Transposes the tensor.
         * 
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the transposed tensor (default: DEFAULTDEVICE).
         * @return The transposed tensor.
         */
        Tensor transpose(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Adds another tensor to this tensor element-wise.
         * 
         * @param other The tensor to add.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return The result of the element-wise addition.
         */
        Tensor add(Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator + (Tensor x) {return add(x);}

        /**
         * @brief Adds a scalar to each element of this tensor.
         * 
         * @param scalar The scalar value to add.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return The result of adding the scalar to each element.
         */
        Tensor add(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator + (double x) {return add(x);}
        friend Tensor operator + (double n, Tensor x) {return x.add(n);}

        /**
         * @brief Subtracts another tensor from this tensor element-wise.
         * 
         * @param other The tensor to subtract.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return The result of the element-wise subtraction.
         */
        Tensor subtract( Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator - (Tensor x) {return subtract(x);}

        /**
         * @brief Subtracts a scalar from each element of this tensor.
         * 
         * @param scalar The scalar value to subtract.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return The result of subtracting the scalar from each element.
         */
        Tensor subtract(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator - (double x) {return subtract(x);}
        friend Tensor operator - (double n, Tensor x) {return fill({1}, n).subtract(x);}

        /**
         * @brief Multiplies this tensor element-wise with another tensor.
         * 
         * @param other The tensor to multiply.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return The result of the element-wise multiplication.
         */
        Tensor elementwiseMult(Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator * (Tensor x) {return elementwiseMult(x);}

        /**
         * @brief Multiplies each element of this tensor by a scalar.
         * 
         * @param scalar The scalar value to multiply.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return The result of multiplying each element by the scalar.
         */
        Tensor elementwiseMult(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator * (double x) {return elementwiseMult(x);}
        friend Tensor operator * (double n, Tensor x) {return x.elementwiseMult(n);}

        /**
         * @brief Divides this tensor element-wise by another tensor.
         * 
         * @param other The tensor to divide by.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return The result of the element-wise division.
         */
        Tensor elementwiseDivision(Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator / (Tensor x) {return elementwiseDivision(x);}

        /**
         * @brief Divides each element of this tensor by a scalar.
         * 
         * @param scalar The scalar value to divide by.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return The result of dividing each element by the scalar.
         */
        Tensor elementwiseDivision(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator / (double x) {return elementwiseDivision(x);}
        friend Tensor operator / (double n, Tensor x) {return fill({1}, n).elementwiseDivision(x);}

        /**
         * @brief Negates the tensor, performing an element-wise negation.
         * 
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return A tensor with negated values.
         */
        Tensor neg(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Raises each element of the tensor to the specified power.
         * 
         * @param exponent The power to which each element will be raised.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return A tensor with each element raised to the specified power.
         */
        Tensor pow(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Applies the ReLU (Rectified Linear Unit) function element-wise.
         * 
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return A tensor after applying ReLU.
         */
        Tensor relu(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Binarizes the tensor, setting all elements greater than 0 to 1 and elements less
         * than or equal to 0 to 0.
         * 
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return A binarized tensor.
         */
        Tensor binarize(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Computes the element-wise reciprocal of the tensor.
         * 
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return A tensor with the reciprocal of each element.
         */
        Tensor reciprocal(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE) {return ones({1}).elementwiseDivision(*this, saveGradient, device);}

        /**
         * @brief Performs matrix multiplication between tensors.
         * 
         * @param other The other tensor to multiply.
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return The result of matrix multiplication.
         */
        Tensor matmul(Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        
        /**
         * @brief Reduces the tensor by summing all its elements.
         * 
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return A tensor with the sum of all elements.
         */
        Tensor reduceSum(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        /**
         * @brief Computes the softmax of the tensor.
         * 
         * @param saveGradient Whether to compute gradients for this operation (default: false).
         * @param device Device to allocate the resulting tensor (default: DEFAULTDEVICE).
         * @return A tensor with the softmax applied.
         */
        Tensor softmax(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE) {return this->elementwiseDivision(this->reduceSum(saveGradient), saveGradient, device);}
};
#endif

