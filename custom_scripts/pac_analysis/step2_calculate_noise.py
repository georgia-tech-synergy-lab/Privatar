import numpy as np
import math


if __name__ == "__main__":
    # bound as 10
    covariance_matrix_z_outsource = np.load("/home/jianming/work/Privatar_prj/custom_scripts/pac_analysis/covariance_matrix_10_z_outsource.npy")
    print(np.linalg.det(covariance_matrix_z_outsource))

    unit_diagonal = np.diag(np.full(covariance_matrix_z_outsource.shape[0],1))
    for b in range(0, 100):
        var = (b + 1) * 0.000001 + 12.308
        result_matrix = unit_diagonal + covariance_matrix_z_outsource  /var
        if abs(np.linalg.det(result_matrix) - math.e**20) < 1e-2:
            print(f"[Done]noise var={var}, diff = {abs(np.linalg.det(result_matrix) - math.e**20)}, det ={np.linalg.det(result_matrix)}")
        else:
            print(f"noise var={var}, math.e**20={math.e**20}, diff = {abs(np.linalg.det(result_matrix) - math.e**20)}, det ={np.linalg.det(result_matrix)}")
