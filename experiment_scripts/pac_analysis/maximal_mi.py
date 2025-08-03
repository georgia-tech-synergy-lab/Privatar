import sympy
from sympy import symbols, ln, diff, limit, oo

# Define the variable x as a symbol
x = symbols('x')

# Define the equation for which we want to find the maximum value
equation = x * ln(65 * x) + (1 - x) * ln(65 * (1 - x) / 64)

# --- Step 1: Find the derivative of the equation with respect to x ---
f_prime = diff(equation, x)
print(f"The derivative of the equation is: {f_prime}\n")

# --- Step 2: Solve for the critical points by setting the derivative to 0 ---
# We solve the equation f'(x) = 0
try:
    critical_points = sympy.solve(f_prime, x)
    print(f"Critical points found at: {critical_points}\n")
except Exception as e:
    critical_points = []
    print(f"Could not solve for critical points automatically: {e}\n")


# --- Step 3: Evaluate the function at the critical points and boundaries [0, 1] ---

# Note: The function is not defined at x=0 and x=1 due to ln(0).
# We must evaluate the limits as x approaches these boundaries.

# Value at the left boundary (limit as x -> 0+)
limit_at_0 = limit(equation, x, 0, dir='+')
print(f"Limit of the function as x approaches 0 from the right: {limit_at_0}")
print(f"Numerical value: {limit_at_0.evalf()}\n")


# Value at the right boundary (limit as x -> 1-)
limit_at_1 = limit(equation, x, 1, dir='-')
print(f"Limit of the function as x approaches 1 from the left: {limit_at_1}")
print(f"Numerical value: {limit_at_1.evalf()}\n")


# Value at the critical points within the interval (0, 1)
max_value = -oo  # Initialize with negative infinity
max_point = None

# Check the boundaries first
if limit_at_0 > max_value:
    max_value = limit_at_0
    max_point = 0

if limit_at_1 > max_value:
    max_value = limit_at_1
    max_point = 1


print("Evaluating at critical points:")
for point in critical_points:
    # Ensure the critical point is within the interval (0, 1)
    if 0 < point < 1:
        value_at_point = equation.subs(x, point)
        print(f"Value at critical point x = {point}: {value_at_point}")
        print(f"Numerical value: {value_at_point.evalf()}\n")
        if value_at_point > max_value:
            max_value = value_at_point
            max_point = point
    else:
        print(f"Critical point x = {point} is outside the interval [0, 1].\n")


# --- Step 4: Determine the maximum value ---
print("---" * 10)
print(f"The maximum value of the equation in the interval [0, 1] is approximately {max_value.evalf()}")
print(f"This occurs as x approaches {max_point}.")

