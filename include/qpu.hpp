#ifndef QPU_HPP
#define QPU_HPP

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <valarray>
#include <random>

#include "tensor.hpp"

namespace qpu
{
	using namespace ds;
	using complexD = std::complex<double>;

	/// @brief A simulation of a Quantum Processing Unit
	class QPU {
	private:
		std::vector<complexD> state;  // Quantum state vector
		std::random_device rd;
		std::mt19937 random_engine{rd()};
	    std::size_t num_qubits;
	public:
		QPU(double alpha, double beta) {
			state[0] = {alpha, 0};
			state[1] = {beta, 0};
		}

		friend std::ostream& operator<<(std::ostream& os, const QPU& qubit) {
			os << "Alpha: " << qubit.state[0] << ", Beta: " << qubit.state[1];
			return os;
		}

		template<class T>
		void applyGate(const T &gateMatrix) {
			// TODO: what operation is done here, simplify it
			std::complex<double> newState[2] = {0, 0};
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					newState[i] += gateMatrix(i,j) * state[j];
				}
			}
			state[0] = newState[0];
			state[1] = newState[1];
		}

		template<class T>
		void applyControlGate(const T& gateMatrix, QPU& control) {
			Tensor<complexD, 2, 2> controlledGateMatrix; // = {{1, 0}, {0, 1}};
			controlledGateMatrix(0,0) = 1; controlledGateMatrix(1,1) = 1;
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					controlledGateMatrix(i,j) = gateMatrix(i,j) * control.state[j];
				}
			}
			applyGate(controlledGateMatrix);
		}

		void applyToffoli(QPU& control1, QPU& control2) {
			Tensor<complexD, 8, 8> toffoliGate; // =  {{1, 0, 0, 0, 0, 0, 0, 0},
													// {0, 1, 0, 0, 0, 0, 0, 0},
													// {0, 0, 1, 0, 0, 0, 0, 0},
													// {0, 0, 0, 1, 0, 0, 0, 0},
													// {0, 0, 0, 0, 1, 0, 0, 0},
													// {0, 0, 0, 0, 0, 1, 0, 0},
													// {0, 0, 0, 0, 0, 0, 0, 1},
													// {0, 0, 0, 0, 0, 0, 1, 0}};

			toffoliGate(0,0) = 1; toffoliGate(1,1) = 1; toffoliGate(2,2) = 1; toffoliGate(3,3) = 1;
			toffoliGate(4,4) = 1; toffoliGate(5,5) = 1; toffoliGate(6,7) = 1; toffoliGate(7,6) = 1;
			
			applyControlGate(toffoliGate, control2);
		}

	template<class T>
    void apply_gate(const T& gate_matrix, const std::vector<int>& target_qubits) {
        // Apply a gate to the specified target qubits
        T gate_product = gate_matrix;

        // Compute the Kronecker product with identity matrices for other qubits
        for (int qubit_idx = 0; qubit_idx < num_qubits; ++qubit_idx) {
            if (std::find(target_qubits.begin(), target_qubits.end(), qubit_idx) == target_qubits.end()) {
                gate_product = kronecker_product(gate_product, T::identity(2));
            }
        }

        // Update the state using the gate product
        state = gate_product * state;
    }

	int measure() {
		double probabilities[2] = {std::norm(state[0]), std::norm(state[1])};
		std::discrete_distribution<int> distribution({probabilities[0], probabilities[1]});
		int outcome = distribution(random_engine);
		state[0] = {0, 0};
		state[1] = {0, 0};
		state[outcome] = {1, 0};
		return outcome;
	}

    std::string measure_() {
        // Perform measurement on the qubits
        std::vector<double> probabilities;
        for (std::size_t i = 0; i < state.size(); i += 2) {
            double prob = std::norm(state[i]) + std::norm(state[i + 1]);
            probabilities.push_back(prob);
        }

        // Choose a measurement outcome based on probabilities
        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
        int measured_index = distribution(random_engine);
        state = std::vector<complexD>(0.0, num_qubits);
        state[measured_index] = 1.0;

        // Convert the measured index to a binary string
        std::string measurement_result;
        int remaining = measured_index;
        for (int i = num_qubits - 1; i >= 0; --i) {
            measurement_result += std::to_string(remaining / (1 << i));
            remaining %= 1 << i;
        }

        return measurement_result;
    }


		using gate22 = Tensor<complexD, 2, 2>;
		using gate44 = Tensor<complexD, 4, 4>;

		// TODO: Adjust Tensor so I can constexpr these -

		/// Hadamard gate is used to create superpositions.
		inline static const gate22 H_gate = {1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0)};
		/// Pauli-X gate is analogous to a classical NOT gate and is often used for flipping qubit states.
		inline static const gate22 X_gate = {0.0, 1.0, 1.0, 0.0};
		/// Pauli-Y gate is similar to the Pauli-X gate but includes complex phases.
		inline static const gate22 Y_gate = {complexD(0,0), complexD(0,-1), complexD(0,1), complexD(0,0)};
		/// Pauli-Z gate introduces a phase shift of -1 to the |1⟩ state (it leaves the |0⟩ state unchanged)
		inline static const gate22 Z_gate = {1.0, 0.0, 0.0, -1.0};
		/// Phase gate introduces a π/2 phase shift to the |1⟩ state (it leaves the |0⟩ state unchanged).
		inline static const gate22 S_gate = {complexD(1,0), complexD(0,0), complexD(0,0), complexD(0,1)};
		/// T gate introduces a π/4 phase shift to the |1⟩ state (it leaves the |0⟩ state unchanged).
		inline static const gate22 T_gate = {complexD(1,0), complexD(0,0), complexD(0,0), complexD(0,std::exp(1.0 * M_PI / 4.0))};
		/// CNOT (Controlled-X) gate is a two-qubit gate that flips the target qubit if and only if the control qubit is in state |1⟩. 
		inline static const gate44 CNOT_gate = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};

	};

} // namespace qpu


#endif // QPU_HPP