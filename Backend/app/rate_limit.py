import os
from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory limiter: suitable for a single process / demo deployment.
# For multiple replicas, replace storage with Redis-backed SlowAPI storage.
# Disabled in test environment to avoid spurious 429 errors during automated testing.
is_test = os.getenv("APP_ENV") == "test"
limiter = Limiter(key_func=get_remote_address, default_limits=[], enabled=not is_test)
