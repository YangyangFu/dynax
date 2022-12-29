import jax
import jax.numpy as jnp
import time 
import os
# for CPU performance
import numpy as np

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
#os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                           "intra_op_parallelism_threads=1")
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

### Compare CPU vs GPU
import numpy as np
import jax.numpy as jnp
import jax

def f(x):  # function we're benchmarking (works in both NumPy & JAX)
  return x.T @ (x - x.mean(axis=0))

x_np = np.ones((1000, 1000), dtype=np.float32)  # same as JAX default dtype
ts_cpu = time.time()
f(x_np)  # measure NumPy runtime
dt_cpu =time.time() - ts_cpu
print(f"Numpy using CPU: {dt_cpu} seconds")

ts_transfer = time.time()
x_jax = jax.device_put(x_np)  # measure JAX device transfer time
dt_transfer = time.time() - ts_transfer
print(f"Jax transfer data: {dt_transfer} seconds")

f_jit = jax.jit(f)
ts_compile = time.time()
f_jit(x_jax).block_until_ready()  # measure JAX compilation time
dt_compile = time.time() - ts_compile
print(f"Jax compile time: {dt_compile} seconds")

ts_gpu = time.time()
f_jit(x_jax).block_until_ready()  # measure JAX runtime
dt_gpu = time.time() - ts_gpu
print(f"Jax using GPU: {dt_gpu} seconds")
print("====================================")
# How about pure calculation without jit
# ===========================================
ts_cpu = time.time()
x_np.T @ (x_np - x_np.mean(axis=0))  # measure NumPy runtime
dt_cpu =time.time() - ts_cpu
print(f"Numpy using CPU: {dt_cpu} seconds")

ts_transfer = time.time()
x_jax = jax.device_put(x_np)  # measure JAX device transfer time
dt_transfer = time.time() - ts_transfer
print(f"Jax transfer data: {dt_transfer} seconds")

f_jit = jax.jit(f)
ts_compile = time.time()
x_jax.T @ (x_jax - x_jax.mean(axis=0))
dt_compile = time.time() - ts_compile
print(f"Jax compile time: {dt_compile} seconds")

ts_gpu = time.time()
x_jax.T @ (x_jax - x_jax.mean(axis=0)) # measure JAX runtime
dt_gpu = time.time() - ts_gpu
print(f"Jax using GPU: {dt_gpu} seconds")