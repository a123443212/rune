.PHONY: cpp rust python triangle smoke clean

cpp:
	bash scripts/cpp_configure.sh
	bash scripts/cpp_build.sh
	bash scripts/cpp_test.sh

rust:
	bash scripts/rust_test.sh

python:
	bash scripts/python_test.sh

triangle:
	bash scripts/cpp_configure.sh
	bash scripts/triangle_build_eval.sh
	bash scripts/rust_build.sh
	bash scripts/positions_head.sh
	bash scripts/triangle_check_gab.sh
	bash scripts/triangle_check_mh4.sh

smoke:
	bash scripts/smoke_pool.sh
	bash scripts/smoke_run.sh

clean:
	bash scripts/clean_build.sh
	bash scripts/clean_target.sh
