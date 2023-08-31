#include <iostream>
#include <complex>
#include <random>
#include <vector>
#include <cmath>

#include "tensor.hpp"
#include "qpu.hpp"

using namespace ds;
using namespace qpu;
using complexD = std::complex<double>;

int main() {
    // std::vector<std::vector<std::complex<double> > > X_gate = {{0, 1}, {1, 0}};
    // std::vector<std::vector<std::complex<double> > > Y_gate = {{0, -1i}, {1i, 0}};
    // std::vector<std::vector<std::complex<double> > > Z_gate = {{1, 0}, {0, -1}};
	Tensor<complexD, 2, 2> X_gate = {0, 1, 1, 0};
	// X_gate(0,0) = 0; X_gate(0,1) = 1; X_gate(1,0) = 1; X_gate(1,1) = 0;

	Tensor<complexD, 2, 2> Y_gate = {complexD(0,0), complexD(0,-1), complexD(0,1), complexD(0,0)};
	// X_gate(0,1).imag(-1); X_gate(1,0).imag(1); //X_gate(0,0) = 0; X_gate(1,1) = 0;
	
	Tensor<complexD, 2, 2> Z_gate = {1, 0, 0, -1};
	// X_gate(0,0).real(1); X_gate(1,1).real(-1); //X_gate(0,1) = 0; X_gate(1,0) = 0;

    QPU qubit(1.0, 0.0);

    std::cout << "Initial qubit state: " << qubit << std::endl;

    qubit.applyGate(X_gate);
    std::cout << "After X-gate: " << qubit << std::endl;

    qubit.applyGate(Y_gate);
    std::cout << "After Y-gate: " << qubit << std::endl;

    qubit.applyGate(Z_gate);
    std::cout << "After Z-gate: " << qubit << std::endl;

    // Control gates and Toffoli gate example
    QPU control1(1.0, 0.0);
    QPU control2(1.0, 0.0);

    std::cout << "Control Qubit 1: " << control1 << std::endl;
    std::cout << "Control Qubit 2: " << control2 << std::endl;

    qubit.applyControlGate(X_gate, control1);
    std::cout << "After X-gate controlled by Control Qubit 1: " << qubit << std::endl;

    qubit.applyToffoli(control1, control2);
    std::cout << "After Toffoli gate: " << qubit << std::endl;

    int measurement_result = qubit.measure();
    std::cout << "Measurement outcome: " << measurement_result << std::endl;

    return 0;
}
