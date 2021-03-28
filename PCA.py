import tensorflow as tf
import copy
from sklearn.decomposition import PCA

class PCA:
    def __init__(self, matrix, compress_dimension):
        self.matrix = tf.constant(matrix)
        self.compress_dimension = compress_dimension
        if compress_dimension > len(matrix[0]):
            print("input error")
        self.calculate_pca()

    def calculate_pca(self):
        #参数归一化
        self.matrix -= tf.reduce_mean(self.matrix, axis=0, keepdims=True)
        #求协方差矩阵
        matrix_transpose = tf.transpose(self.matrix)
        cov_matrix = tf.matmul(matrix_transpose, self.matrix) / self.matrix.shape[0]
        #求协方差矩阵的特征向量于对角矩阵
        self.diagonal_matrix = tf.linalg.diag(cov_matrix)
        self.eigenvalues, self.eigenvectors = tf.linalg.eigh(cov_matrix)
        #求解pca
        value_list = self.eigenvalues.get_shape().as_list()
        vector_list = self.eigenvectors.get_shape().as_list()
        top_value_vector = []
        for i in range(self.compress_dimension):
            max_index = value_list.index(max(value_list))
            top_value_vector.append(vector_list[max_index])
            value_list[max_index] = min(value_list) - 1
        top_value_vector_tensor = tf.constant(top_value_vector)
        self.pca = tf.matmul(top_value_vector_tensor, self.matrix)
        self.pca = tf.transpose(self.pca)


    def get_result(self):
        return self.diagonal_matrix, self.eigenvalues, self.eigenvectors, self.pca


class PCA_Sklearn:
    def __init__(self, matrix, compress_dimension):
        self.matrix = matrix
        self.compress_dimension = compress_dimension
        self.calculate()

    def calculate(self):
        pca = PCA(n_components=self.compress_dimension)
        self.compressed_data = pca.fit_transform(self.matrix)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

    def get_result(self):
        return self.compressed_data, self.explained_variance_ratio_






