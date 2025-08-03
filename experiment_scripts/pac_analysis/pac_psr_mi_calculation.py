import sympy
from sympy import symbols, ln, nsolve

# Define the variable x as a symbol
x = symbols('x')

mi_list = [4, 3, 1, 0.1, 0.01]
for mi in mi_list:
  # Define the equation
  # The equation is x*ln(65*x) + (1-x)*ln(65*(1-x)/64) = 1
  # We rewrite it as f(x) = 0
  equation = x * ln(65 * x) + (1 - x) * ln(65 * (1 - x) / 64) - mi

  # Provide an initial guess for the solver.
  # Since the arguments of the logarithms must be positive,
  # 65*x > 0  => x > 0
  # 65*(1-x)/64 > 0 => 1-x > 0 => x < 1
  # So, the solution must be in the interval (0, 1).
  # We can choose an initial guess in this interval, for example, 0.5.
  if mi > 5:
    initial_guess = 10
  else:
    initial_guess = 0.5

  # Use nsolve to find the numerical solution
  try:
      solution = nsolve(equation, x, initial_guess)
      print(f"The solution for x is: {solution}")
  except Exception as e:
      print(f"Could not find a solution. Error: {e}")
      # If the initial guess doesn't work, we can try another one.
      # For example, let's try a value closer to 0.
      try:
          initial_guess_2 = 0.1
          solution = nsolve(equation, x, initial_guess_2)
          print(f"The solution for x with a different initial guess is: {solution}")
      except Exception as e2:
          print(f"Could not find a solution with the second guess either. Error: {e2}")

