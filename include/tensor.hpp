#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <valarray>
#include <array>

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <valarray>
#include <random>

namespace ds //Data Structures
{
	template<typename T, size_t... Dims>
	class Tensor {
	public:
	// TODO: add spport for initializer list and infer the dimension from the list
		// Tensor() = default;
		constexpr Tensor() : size_(computeSize()), data_(std::valarray<T>(T{}, size_)) {}

		constexpr Tensor(std::initializer_list<T> init) : size_(computeSize()), data_(init) {}

		template<typename U>
		constexpr Tensor(const Tensor<U, Dims...>& other) : size_(computeSize()), data_(other.size()) {
			data_(other.data_);
		}

		template<typename... Args>
		T& operator()(Args... args) {
			static_assert(sizeof...(args) == sizeof...(Dims), "Invalid number of indices");
			std::array<size_t, sizeof...(Dims)> indices = { static_cast<size_t>(args)... };
			size_t idx = computeIndex(indices);
			return data_[idx];
		}

		template<typename... Args>
		constexpr const T& operator()(Args... args) const {
			static_assert(sizeof...(args) == sizeof...(Dims), "Invalid number of indices");
			std::array<size_t, sizeof...(Dims)> indices = { static_cast<size_t>(args)... };
			size_t idx = computeIndex(indices);
			return data_[idx];
		}

		size_t size() const {
			return size_;
		}

		// Tensor scalar multiplication, returns a new tensor
		template<typename U>
		Tensor<T, Dims...> operator*(const U& value) const {
			Tensor<T, Dims...> result(*this);
			result.data_ *= value;
			return result;
		}

		// Add two tensors and return a new tensor
		Tensor<T, Dims...> operator+(const Tensor<T, Dims...>& other) const {
			Tensor<T, Dims...> result(*this);
			result.data_ += other.data_;
			return result;
		}

		// template<typename T, size_t... Dims>
		size_t getDimension(const Tensor<T, Dims...>& tensor) {
			// static_assert(sizeof...(Dims) > 0, "Tensor must have at least one dimension");
			return sizeof...(Dims);
		}

	private:
		size_t size_{0};
		std::valarray<T> data_{};

		constexpr size_t computeSize() const {
			if constexpr (sizeof...(Dims) > 0) {
				return (Dims * ...);
			}

			if constexpr (sizeof...(Dims) == 0) {
				return 1;
			}
			return 0;
		}

		// index = indices[0] * stride[0] + indices[1] * stride[1] + ... + indices[N-1] * stride[N-1]
		template<size_t... Is>
		constexpr size_t computeIndex(const std::array<size_t, sizeof...(Dims)>& indices, std::index_sequence<Is...>) const {
			if constexpr (sizeof...(Is) > 0) {
				return ((computeStride<Is>() * indices[Is]) + ...);
			}
			return 0;
		}

		constexpr size_t computeIndex(const std::array<size_t, sizeof...(Dims)>& indices) const {
			return computeIndex(indices, std::make_index_sequence<sizeof...(Dims)>{});
		}

		// stride[I] = stride[I+1] * dimension[I+1],; where stride[I] = 1 if I+1 >= sizeof...(Dims)
		template<size_t I>
		constexpr size_t computeStride() const {
			// size_t stride = 1;
			// ((I > 0 && (stride *= Dims - I + 1)), ...);

			if constexpr (I + 1 < sizeof...(Dims)) {
				size_t stride = 1;
				// ((stride *= getDimension<I+1>()), ...);

				// ([&](auto dim){std::cout << dim << ", ";} (Dims), ...);
				// // auto printor = [&](auto dim){std::cout << dim << ", ";};
				// // // fold over a comma operator
				// // (printor(Dims), ...);

				std::array<size_t, sizeof...(Dims)> dimensions{Dims...};
				for (size_t j = I + 1; j < dimensions.size(); ++j) {
					stride *= dimensions[j];
				}
				return stride;
			}
			return 1;
		}

		template <size_t I>
		constexpr size_t getDimension() const {
			static_assert(I < sizeof...(Dims), "Invalid dimension index");
			return std::get<I>(std::array<size_t, sizeof...(Dims)>{Dims...});
		}

	};

} // namespace ds

#endif // TENSOR_HPP