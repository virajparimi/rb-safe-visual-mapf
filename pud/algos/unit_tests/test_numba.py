import unittest
import numpy as np
from numba import types
from numba import jit, njit
from numba.typed import Dict

"""
python pud/algos/unit_tests/test_numba.py TestNumba.test_nested_dicts
"""


class TestNumba(unittest.TestCase):
    def test_basic_f(self):
        @jit(nopython=True)
        def basic_f(x, y):
            return x + y

        _ = basic_f(2, 3)

    def test_f_w_dict(self):
        @jit(nopython=True)
        def f_w_dict(x):
            total = 0
            for key in x:
                total = total + x[key]
            return total

        inp = {
            "x": 1,
            "y": 2,
        }
        out = f_w_dict(inp)
        self.assertTrue(out == 3)

    def test_dict_example(self):
        # First create a dictionary using Dict.empty()
        # Specify the data types for both key and value pairs

        # Dict with key as strings and values of type float array
        dict_param1 = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64[:],
        )

        # Dict with keys as string and values of type float
        dict_param2 = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64,
        )

        # Type-expressions are currently not supported inside jit functions.
        float_array = types.float64[:]

        @njit
        def add_values(d_param1, d_param2):
            # Make a result dictionary to store results
            # Dict with keys as string and values of type float array
            result_dict = Dict.empty(
                key_type=types.unicode_type,
                value_type=float_array,
            )

            for key in d_param1.keys():
                result_dict[key] = d_param1[key] + d_param2[key]

            return result_dict

        dict_param1["hello"] = np.asarray([1.5, 2.5, 3.5], dtype="f8")
        dict_param1["world"] = np.asarray([10.5, 20.5, 30.5], dtype="f8")

        dict_param2["hello"] = 1.5
        dict_param2["world"] = 10

        final_dict = add_values(dict_param1, dict_param2)

        print(final_dict)
        # Output : {hello: [3. 4. 5.], world: [20.5 30.5 40.5]}

    def test_nested_dicts(self):
        """
        TODO: Seems the input are all numba typed, is this required?
        """
        from numba import njit, typeof, typed, types

        d1 = typed.Dict.empty(
            key_type=types.float64,
            value_type=types.float64,
        )
        d2 = typed.Dict.empty(
            key_type=types.float64,
            value_type=typeof(d1),  # Base the d2 instance values of the type of d1
        )

        print("d1's Numba type is", typeof(d1))
        # d1 is an instance so you can use it like a dict
        d1[5.0] = 6.0
        print(d1)

        @njit
        def foo(d2):
            d2[1.0] = {2.0: 3.0}
            return d2

        print("Using d2")
        print(foo(d2))

        # You can also spell it like this:
        d1_type = types.DictType(types.float64, types.float64)
        d3 = typed.Dict.empty(types.float64, d1_type)

        print("Using d3")
        print(foo(d3))


if __name__ == "__main__":
    unittest.main()
