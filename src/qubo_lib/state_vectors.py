import numpy as np
from typing import Optional


class StateVectors:
    """
    Container for binary state vectors in packed format.
    
    Efficiently stores multiple binary state vectors using bitwise packing.
    Provides methods for converting between wide and packed representations.
    """

    SUPPORTED_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64}
    
    def __init__(self, wide_vector_matrix: np.ndarray):
        """
        Initialize StateVectors from wide-format binary matrix.
        
        Args:
            wide_vector_matrix (np.ndarray): Binary matrix of shape (samples_cnt, N) 
                                           where each row is a binary state vector
        """
        self.N = wide_vector_matrix.shape[1]
        self.element_bits = 8
        self.bitwise_vectors = self.pack_wide_vectors_to_bitwise(wide_vector_matrix)

    @classmethod
    def from_packed_vectors(cls, packed_vector_matrix: np.ndarray, N: int) -> 'StateVectors':
        if packed_vector_matrix.dtype.type not in cls.SUPPORTED_TYPES:            
            raise TypeError(f"Unsupported dtype: {packed_vector_matrix.dtype}. Supported types: {cls.SUPPORTED_TYPES}")
        
        instance = cls.__new__(cls)  # Создаем экземпляр без вызова __init__
        instance.bitwise_vectors = packed_vector_matrix
        instance.N = N
        instance.element_bits = packed_vector_matrix.dtype.itemsize * 8
        return instance

    def pack_wide_vectors_to_bitwise(self, wide_vectors: np.ndarray, target_dtype=np.uint8) -> np.ndarray:
        if target_dtype not in self.SUPPORTED_TYPES:
            raise TypeError(f"Unsupported target dtype: {target_dtype}. Supported types: {self.SUPPORTED_TYPES}")
            
        num_vectors, vector_length = wide_vectors.shape
        
        bits_per_element = np.dtype(target_dtype).itemsize * 8
        elements_per_vector = (vector_length + bits_per_element - 1) // bits_per_element
        
        bitwise_vectors = np.zeros((num_vectors, elements_per_vector), dtype=target_dtype)
        
        for i in range(num_vectors):
            for j in range(vector_length):
                if wide_vectors[i, j]:
                    element_index = j // bits_per_element
                    bit_index = j % bits_per_element
                    bitwise_vectors[i, element_index] |= (1 << bit_index)
        
        return bitwise_vectors

    def get_wide_vectors(self) -> np.ndarray:
        """
        Get stored state vectors as a wide-format binary matrix.
        
        Returns:
            np.ndarray: Binary matrix of shape (samples_cnt, N) where each row 
                       is a binary state vector in wide format (0/1 values)
        """
        if self.bitwise_vectors is None:
            return np.array([])
            
        num_vectors, elements_per_vector = self.bitwise_vectors.shape
        
        wide_vectors = np.zeros((num_vectors, self.N), dtype=np.uint8)
        
        for i in range(num_vectors):
            for j in range(self.N):
                element_index = j // self.element_bits
                bit_index = j % self.element_bits
                if self.bitwise_vectors[i, element_index] & (1 << bit_index):
                    wide_vectors[i, j] = 1
        
        return wide_vectors