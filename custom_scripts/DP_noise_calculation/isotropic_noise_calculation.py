import math
import numpy as np

l2_norm_outsourced_data = 3420
posterious_successful_rate_list = [0.4, 0.09]
prior_successful_rate = 1/56
dimensionality = 256

def main():
  """
  [1] https://arxiv.org/pdf/1702.07476**

  First, calculate \( \epsilon = \frac{\alpha}{2 \sigma^2} \).

  Where \( y = \text{Posterior Successful Rate (PSR)} \),

  \( k = \text{Prior Successful Rate} \) (under the expression identification attack, it is \( \frac{1}{65} \)).

  In Theorem 2, \( \epsilon = \frac{\alpha c^2}{2 \sigma^2} \), where \( c^2 = \text{profiled \( L_2 \) norm (260)} \).

  Substitute \( \epsilon \), \( y \), and \( k \) into Theorem 2 to obtain an equation involving \( \alpha \) and \( \sigma \).

  In this equation, iterate through \( \alpha = 1 \) to \( 20 \), calculate \( \sigma \), and select the smallest value of \( \sigma \).

  sigma^2 = alpha*c^2/[2*alpha/(alpha-1)*ln{pri} - 2*ln{PSR}]

  For the PAC experiment, we used the following configurations:

  1. **MI = 1 (PSR = 0.40)**  
  2. **MI = 0.1 (PSR = 0.09)**  
  3. **MI = 0.01 (PSR = 0.035)**  
  """

  c_square = l2_norm_outsourced_data * l2_norm_outsourced_data
  kappa = prior_successful_rate
  result_sigma_list = []
  for gamma in posterious_successful_rate_list:
    minimal_sigma = 9999999999
    for alpha in range(2,20):
      divisor = (2*alpha/(alpha-1)*math.log(gamma) - 2*math.log(kappa))
      if divisor > 0:
        sigma_square = alpha*c_square/divisor
        if sigma_square < minimal_sigma:
          minimal_sigma = sigma_square
    result_sigma_list.append(minimal_sigma)

  return result_sigma_list

if __name__ == "__main__":
  result_sigma_list = main()
  for i, psr in enumerate(posterious_successful_rate_list):
    print(f"L2 norm = {math.sqrt(result_sigma_list[i])*dimensionality}, minimal sigma = {math.sqrt(result_sigma_list[i])} for Posterious Successful Rate = {psr}")
    # print(f"L2 norm = {result_sigma_list[i]*dimensionality}, minimal sigma^2 = {result_sigma_list[i]} for Posterious Successful Rate = {psr}")

