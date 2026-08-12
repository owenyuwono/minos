# minos — shortcuts for the things you actually launch.
# Needs cargo on PATH and a Vulkan-capable GPU. Bare `make` starts the planet.

.PHONY: run release viewer classic check test clean help

# The planet: voxel terrain + flora. This is the default build.
run:
	cargo run -p minos-app

# Same thing, optimized. The startup tectonics bake is ~18s debug vs ~1-3s here,
# so use this whenever you are not stepping through the app.
release:
	cargo run -p minos-app --release

# Standalone procedural-tree viewer (the flora showcase).
viewer:
	cargo run -p minos-app --bin flora_viewer --features flora

# Classic quadtree engine — no voxel terrain, no flora.
classic:
	cargo run -p minos-app --no-default-features

# Compile without linking. Use this while the app is running: the .exe is locked,
# so a `build` would fail where `check` succeeds.
check:
	cargo check -p minos-app

test:
	cargo test --workspace

clean:
	cargo clean

help:
	@echo Targets:
	@echo   make run      - start the planet, debug build
	@echo   make release  - start the planet, optimized
	@echo   make viewer   - standalone procedural-tree viewer
	@echo   make classic  - quadtree engine, no voxel or flora
	@echo   make check    - compile check, safe while the app runs
	@echo   make test     - full workspace test suite
	@echo   make clean    - delete target/
