from dynax.wrapper.spaces import Space, Discrete, Box 
import jax
import jax.numpy as jnp


def test_sample_rng(space:Space):
    key = jax.random.PRNGKey(0)
    sample1 = space.sample(key)
    sample2 = space.sample(key)

    assert jnp.all(sample1 == sample2)
    
    new_key, _ = jax.random.split(key)
    sample3 = space.sample(new_key)
    assert jnp.any(sample1 != sample3)


def test_space_equality(space1:Space, space2:Space):

    assert space1 == space1
    assert space2 == space2 
    assert space1 != space2, f"space 1 should not equal to space 2, space 1:{space1}, space 2: {space2}"


def test_contains(space:Space):
    key = jax.random.PRNGKey(0)
    for _ in range(10):
        key, _ = jax.random.split(key)
        sample = space.sample(rng=key)

        assert space.contains(sample)
    
if __name__ == "__main__":
    
    discrete1 = Discrete(
        n = 6,
        start = 1,
    )

    discrete2 = Discrete(
        n = 6,
        start = 0,
    )

    box1 = Box(
        low=jnp.array([1,2,3,4]),
        high=jnp.array([5,6,7,8]),
        shape=None,
        dtype=jnp.int16)

    box2 = Box(
        low=jnp.array([1,2,3,4]),
        high=jnp.array([5,6,7,8]),
        shape=None,
        dtype=jnp.float16)
    
    spaces = [
        discrete1, discrete2, box1, box2
    ]

    # test sample rng
    for space in spaces:
        test_sample_rng(space)
    
    # test space equality
    for i in range(len(spaces)-1):
        test_space_equality(spaces[i], spaces[i+1])
    
    # test contains
    for space in spaces:
        test_contains(space)

    print("all tests passed in wrapper.spaces!!!")