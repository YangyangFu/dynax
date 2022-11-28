import jax
import jax.numpy as jnp
import time 
import os
# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

s = time.time()
key = jax.random.PRNGKey(42)
x = jax.random.normal(key=key, shape=(50000, 50000))
x @ x
e = time.time()

print("========")
print(f"execution time is: {e-s} seconds!!")

