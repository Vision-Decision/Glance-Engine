# Scripts that share the test_ prefix but are CLI suites, not pytest modules. The CPU
# equivalence suite runs through tests/test_equivalence_cpu.py; the GPU one is documented
# in docs/exactness.md.
collect_ignore = ["scan/test_scan_equivalence.py", "speed/test_scheduler_gpu.py", "scan/vm", "demo"]
