# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Benchmark suites. Each module registers its `Bench` instances with
`runner.register(...)` at import time. The runner imports the modules
listed in `BenchConfig.suites`."""
