# ml-inference Makefile
#
# Targets:
#   make host              Build + test for desktop Linux
#   make host-no-et        Build without ExecuTorch (beam search + resampler only)
#   make android           Build for all Android ABIs (arm64-v8a, armeabi-v7a, x86_64, x86)
#   make android-arm64-v8a Build for a single ABI
#   make push              Push binary + encoder/vocab/layout/libc++ to device
#   make push-demo         Push all demo files (models, layouts, vocabs)
#   make push-eval         Push infer_jsonl + eval_accuracy + eval datasets
#   make benchmark         Build, push, run on-device benchmark
#   make export-bench      Run benchmark + write benchmark.json outputs
#   make eval-jsonl-android Run JSONL accuracy eval on device (fp16 pipeline)
#   make eval-futo-android  Run futo_test.bin eval on device (fp16 pipeline)
#   make test              Run host unit tests
#   make test-sanity       Run model-backed engine sanity tests
#   make test-regression   Run host-side accuracy regression (pass/fail thresholds)
#   make test-android      Run Kotlin instrumented tests on device
#   make clean             Remove all build artifacts (host, android, asan)
#   make clean-host        Remove host build only
#   make clean-android     Remove Android build only
#   make clean-asan        Remove ASAN build only
#   make clean-et          Remove ExecuTorch builds (forces full rebuild)
#   make clean-all         Remove everything (build artifacts + ExecuTorch builds)

JOBS          := $(shell nproc 2>/dev/null || echo 4)
JOBS          := $(shell [ $(JOBS) -gt 8 ] && echo 8 || echo $(JOBS))

ET_ROOT       := third_party/executorch
ET_HOST       := $(ET_ROOT)/cmake-out-host
DEVICE_DIR    := /data/local/tmp/ml-inference
BENCH_AFFINITY ?= big

# --- Android ABI support ---
ANDROID_ABIS  := arm64-v8a armeabi-v7a x86_64 x86

# Python with torchgen (ships inside PyTorch), needed for ET selective-op codegen.
# Without a valid PYTHON_EXECUTABLE, cmake's file(GLOB) expands /*.py and hangs.
# A local venv is auto-created with the exact torch version pinned by ET v1.2.0.
VENV          := $(CURDIR)/.venv
VENV_PYTHON   := $(VENV)/bin/python3
VENV_CMAKE    := $(VENV)/bin/cmake
TORCH_VERSION := 2.10.0
TORCHGEN_PYTHON = $(VENV_PYTHON)

# --- NDK detection ---
NDK           := $(or $(ANDROID_NDK_HOME),$(NDK_HOME),$(shell ls -d ~/Android/Sdk/ndk/*/ 2>/dev/null | sort -V | tail -1 | sed 's:/*$$::'))
TOOLCHAIN     := $(NDK)/build/cmake/android.toolchain.cmake

# --- Selective op stripping (auto-detected from models/*/ops.txt) ---
# Comma-separated op list per ET docs:
# https://docs.pytorch.org/executorch/stable/kernel-library-selective-build.html
OPS_FILES := $(wildcard models/*/ops.txt)
ifneq ($(OPS_FILES),)
    SELECTED_OPS := $(shell cat $(OPS_FILES) | sort -u | grep -v '^$$' | paste -sd, -)
endif

# --- Model selection (universal-swipe architecture) ---
ENCODER        ?= honorable_sturgeon
LM_MODEL       ?= hungry_jellyfish
ENCODER_FILE   ?= model_fp32.pte
LM_MODEL_FILE  ?= context_lm.pte
LM_VOCAB_FILE  ?= vocab.txt
# Set DECODER= (empty) to run encoder-only tests.
DECODER        ?= magic_macaw
DECODER_FILE   ?= model_fp32.pte
LAYOUT_FILE    ?= layouts/en_qwerty.json
VOCAB_FILE     ?= vocabs/en.combined

# --- Consistency/eval settings ---
EVAL_DATASET_BIN ?= eval/futo_test.bin
JSONL_DATASET ?= eval/swipe2_test.jsonl
EVAL_LIMIT ?= 300
JSONL_LIMIT ?= 300
CONSIST_TOP_K ?= 4
CONSIST_BEAM_WIDTH ?= 100

# --- Regression thresholds (on the first $(*_LIMIT) samples, en_qwerty) ---
# Floors, not targets: drift below these trips the test.  Tune down only after
# independently verifying the drop is real and accepted, not a regression.
# Encoder-only is the universal floor; decoder/LM should do strictly better.
# Thresholds represent the floor of current performance on canonical assets.
REGRESSION_JSONL_ENC_TOP1 ?= 80
REGRESSION_JSONL_ENC_TOP4 ?= 92
REGRESSION_JSONL_DEC_TOP1 ?= 85
REGRESSION_JSONL_DEC_TOP4 ?= 95
REGRESSION_BIN_ENC_TOP1   ?= 92
REGRESSION_BIN_ENC_TOP4   ?= 98

# --- ExecuTorch CMake flags (shared between host and android) ---
ET_CMAKE_FLAGS = \
	-DPYTHON_EXECUTABLE=$(TORCHGEN_PYTHON) \
	-DEXECUTORCH_BUILD_XNNPACK=ON \
	-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
	-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
	-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
	-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
	-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
	-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
	-DEXECUTORCH_BUILD_VULKAN=OFF \
	-DEXECUTORCH_BUILD_MPS=OFF \
	-DEXECUTORCH_BUILD_COREML=OFF \
	-DEXECUTORCH_BUILD_QNN=OFF \
	-DEXECUTORCH_BUILD_SDK=OFF \
	-DEXECUTORCH_BUILD_TESTS=OFF \
	-DEXECUTORCH_BUILD_EXAMPLES=OFF \
	-DEXECUTORCH_OPTIMIZE_SIZE=ON \
	-DXNNPACK_BUILD_ALL_MICROKERNELS=OFF \
	-DXNNPACK_ENABLE_SPARSE=OFF \
	-DXNNPACK_BUILD_TESTS=OFF \
	-DXNNPACK_BUILD_BENCHMARKS=OFF \
	-DXNNPACK_ENABLE_ARM_SME=OFF \
	-DXNNPACK_ENABLE_ARM_SME2=OFF

ifneq ($(SELECTED_OPS),)
ET_CMAKE_FLAGS += -DEXECUTORCH_SELECT_OPS_LIST="$(SELECTED_OPS)"
endif

# ExecuTorch v1.2.0 bug: XNNWeightsCache.cpp uses try/catch but
# EXECUTORCH_OPTIMIZE_SIZE sets -fno-exceptions globally.  Re-enable
# exceptions for the xnnpack_backend target (same pattern as Vulkan,
# Qualcomm, AOTI, CUDA, MediaTek, and Apple Metal backends).
ET_PATCHES := patches/executorch-xnnpack-fexceptions.patch

patch-et: $(ET_ROOT)/CMakeLists.txt
	@for p in $(ET_PATCHES); do \
		if git -C $(ET_ROOT) apply --reverse --check "../../$$p" >/dev/null 2>&1; then \
			:; \
		elif git -C $(ET_ROOT) apply --check "../../$$p" >/dev/null 2>&1; then \
			echo "=== Applying $$p ==="; \
			git -C $(ET_ROOT) apply "../../$$p"; \
		else \
			echo "Error: $$p neither applies cleanly nor is already applied" >&2; \
			exit 1; \
		fi; \
	done
	sed -i "s/static char hexdigits\\[16\\] = \"0123456789ABCDEF\";/static char hexdigits[17] = \"0123456789ABCDEF\";/" third_party/executorch/third-party/flatcc/include/flatcc/portable/grisu3_print.h

unpatch-et:
	@for p in $(ET_PATCHES); do \
		if git -C $(ET_ROOT) apply --reverse --check "../../$$p" >/dev/null 2>&1; then \
			echo "=== Reverting $$p ==="; \
			git -C $(ET_ROOT) apply --reverse "../../$$p"; \
		fi; \
	done
	sed -i "s/static char hexdigits\\[17\\] = \"0123456789ABCDEF\";/static char hexdigits[16] = \"0123456789ABCDEF\";/" third_party/executorch/third-party/flatcc/include/flatcc/portable/grisu3_print.h

# ===========================================================================
# Python venv (auto-created for ET codegen)
# ===========================================================================

$(VENV_PYTHON):
	@python3 -c "import ensurepip" 2>/dev/null || { echo "Error: python3-venv not installed. Run: sudo apt install python3-venv" >&2; exit 1; }
	@echo "=== Creating venv with torchgen + cmake (for ET codegen) ==="
	python3 -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip cmake pyyaml typing_extensions
	@echo "Extracting torchgen from torch $(TORCH_VERSION) wheel..."
	$(VENV_PYTHON) -m pip download --no-deps torch==$(TORCH_VERSION) \
		--index-url https://download.pytorch.org/whl/cpu -d $(VENV)/tmp
	$(VENV_PYTHON) -c "import zipfile,glob,sysconfig; \
		whl=glob.glob('$(VENV)/tmp/torch-*.whl')[0]; \
		sp=sysconfig.get_path('purelib'); \
		zf=zipfile.ZipFile(whl); \
		[zf.extract(n,sp) for n in zf.namelist() if n.startswith('torchgen/')]"
	rm -rf $(VENV)/tmp

# ===========================================================================
# Host
# ===========================================================================

.PHONY: patch-et unpatch-et host host-no-et test bench test-sanity test-asan test-valgrind lint \
	verify test-consistency test-regression

# Prevent make from deleting these on Ctrl+C (ET takes ~10 min to build)
.PRECIOUS: $(ET_HOST)/libexecutorch.a

$(ET_HOST)/libexecutorch.a: $(ET_ROOT)/CMakeLists.txt | $(VENV_PYTHON) patch-et
	@test -f $(ET_ROOT)/CMakeLists.txt || { echo "Error: ExecuTorch submodule not initialized. Run: git submodule update --init --recursive" >&2; exit 1; }
	@echo "=== Building ExecuTorch (host) ==="
	$(VENV_CMAKE) -B $(ET_HOST) -S $(ET_ROOT) -DCMAKE_BUILD_TYPE=Release $(ET_CMAKE_FLAGS)
	$(VENV_CMAKE) --build $(ET_HOST) -j$(JOBS)

build/libswipe_decoder.a: $(ET_HOST)/libexecutorch.a CMakeLists.txt $(wildcard src/*.cpp include/swipe_decoder/*.hpp tests/*.cpp targets/*.cpp benchmark/*.cpp)
	cmake -B build -DCMAKE_BUILD_TYPE=Release .
	cmake --build build -j$(JOBS)

host: build/libswipe_decoder.a
	@echo "Build complete: build/"
	@ls -la build/libswipe_decoder.a build/test_beam_search build/test_vocab_hash build/test_engine build/benchmark_e2e 2>/dev/null || true

host-no-et: CMakeLists.txt $(wildcard src/*.cpp include/swipe_decoder/*.hpp)
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DET_BUILD=/dev/null .
	cmake --build build -j$(JOBS)
	@echo "Build complete (no ExecuTorch): build/"

test: host
	build/test_beam_search
	build/test_vocab_hash
	build/test_engine

bench: host
	build/test_vocab_hash --bench

test-asan: $(ET_HOST)/libexecutorch.a
	cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DSANITIZE=ON .
	cmake --build build-asan -j$(JOBS)
	build-asan/test_beam_search
	build-asan/test_vocab_hash
	build-asan/test_engine

test-valgrind: host
	valgrind --leak-check=full --error-exitcode=1 build/test_vocab_hash
	valgrind --leak-check=full --error-exitcode=1 build/test_beam_search
	valgrind --leak-check=full --error-exitcode=1 build/test_engine

# Model-backed engine sanity: runs all four inference modes when assets are
# present (encoder, encoder+LM, encoder+decoder, encoder+decoder+LM)
test-sanity: host
	@ENCODER_PTE=models/$(ENCODER)/$(ENCODER_FILE); \
	VOCAB=models/$(VOCAB_FILE); \
	LAYOUT=models/$(LAYOUT_FILE); \
	ALT_LAYOUT=models/layouts/fr_azerty.json; \
	MISMATCH_LAYOUT=models/layouts/ru_east_slavic.json; \
	LM_PTE=models/$(LM_MODEL)/$(LM_MODEL_FILE); \
	LM_VOCAB=models/$(LM_MODEL)/${LM_VOCAB_FILE}; \
	DECODER_PTE=""; \
	if [ -n "$(DECODER)" ]; then \
		if [ ! -f models/$(DECODER)/$(DECODER_FILE) ]; then \
			echo "Error: DECODER=$(DECODER) set but models/$(DECODER)/$(DECODER_FILE) is missing." >&2; \
			echo "       Drop DECODER= to run encoder-only tests." >&2; \
			exit 1; \
		fi; \
		DECODER_PTE=models/$(DECODER)/$(DECODER_FILE); \
	fi; \
	if [ -f "$$ENCODER_PTE" ] && [ -f "$$VOCAB" ] && [ -f "$$LAYOUT" ]; then \
		build/test_engine "$$ENCODER_PTE" "$$VOCAB" "$$LAYOUT" "$$ALT_LAYOUT" "$$LM_PTE" "$$LM_VOCAB" "$$DECODER_PTE" "$$MISMATCH_LAYOUT"; \
	else \
		echo "Skipping test_engine sanity: missing encoder/vocab/layout files"; \
	fi; \
	if [ -f "$$LM_PTE" ] && [ -f "$$VOCAB" ] && [ -f "$$LM_VOCAB" ]; then \
		build/test_lm_vocab "$$LM_PTE" "$$LM_VOCAB" "$$VOCAB"; \
	else \
		echo "Skipping LM vocab coverage: missing context LM or vocab files"; \
	fi

test-consistency:
	@ENCODER_PTE=models/$(ENCODER)/$(ENCODER_FILE); \
	VOCAB=models/$(VOCAB_FILE); \
	LAYOUT=models/$(LAYOUT_FILE); \
	if [ -f "$$ENCODER_PTE" ] && [ -f "$$VOCAB" ] && [ -f "$$LAYOUT" ] && [ -f "$(EVAL_DATASET_BIN)" ]; then \
		echo "=== eval_accuracy ==="; \
		build/eval_accuracy --encoder "$$ENCODER_PTE" --vocab "$$VOCAB" --layout "$$LAYOUT" \
			--dataset "$(EVAL_DATASET_BIN)" \
			--beam-width $(CONSIST_BEAM_WIDTH) --top-k $(CONSIST_TOP_K) \
			--expansion --limit $(EVAL_LIMIT); \
	else \
		echo "Skipping eval_accuracy: missing encoder/vocab/layout/dataset"; \
	fi
	@ENCODER_PTE=models/$(ENCODER)/$(ENCODER_FILE); \
	VOCAB=models/$(VOCAB_FILE); \
	LAYOUT=models/$(LAYOUT_FILE); \
	if [ -f "$$ENCODER_PTE" ] && [ -f "$$VOCAB" ] && [ -f "$$LAYOUT" ] && [ -f "$(JSONL_DATASET)" ]; then \
		echo "=== infer_jsonl ($(JSONL_LIMIT) lines) ==="; \
		TMP_JSONL=$$(mktemp); \
		TMP_OUT=$$(mktemp); \
		head -n $(JSONL_LIMIT) "$(JSONL_DATASET)" > "$$TMP_JSONL"; \
		if ! build/infer_jsonl --encoder "$$ENCODER_PTE" --vocab "$$VOCAB" --layout "$$LAYOUT" \
			--jsonl "$$TMP_JSONL" \
			--beam-width $(CONSIST_BEAM_WIDTH) --top-k $(CONSIST_TOP_K) --expansion > "$$TMP_OUT"; then \
			cat "$$TMP_OUT"; rm -f "$$TMP_OUT" "$$TMP_JSONL"; exit 1; \
		fi; \
		sed -n '/^Results:/,$$p' "$$TMP_OUT"; \
		rm -f "$$TMP_OUT" "$$TMP_JSONL"; \
	else \
		echo "Skipping infer_jsonl: missing encoder/vocab/layout/jsonl"; \
	fi

verify: test-sanity test-consistency

# CI-friendly accuracy regression test.  Runs the canonical eval assets
# (eval/swipe2_test.jsonl + eval/futo_test.bin) in the modes we ship today
# (encoder-only, encoder+decoder when DECODER is set) and fails if top-1 /
# top-4 drops below the REGRESSION_* floors above.
test-regression: host
	@ENCODER_PTE=models/$(ENCODER)/$(ENCODER_FILE); \
	VOCAB=models/$(VOCAB_FILE); \
	LAYOUT=models/$(LAYOUT_FILE); \
	DEC_ARGS=""; DEC_LABEL="encoder-only"; \
	JSONL_TOP1=$(REGRESSION_JSONL_ENC_TOP1); \
	JSONL_TOP4=$(REGRESSION_JSONL_ENC_TOP4); \
	if [ -n "$(DECODER)" ]; then \
		if [ ! -f models/$(DECODER)/$(DECODER_FILE) ]; then \
			echo "Error: DECODER=$(DECODER) set but models/$(DECODER)/$(DECODER_FILE) is missing." >&2; \
			exit 1; \
		fi; \
		DEC_ARGS="--decoder models/$(DECODER)/$(DECODER_FILE)"; \
		DEC_LABEL="encoder+decoder"; \
		JSONL_TOP1=$(REGRESSION_JSONL_DEC_TOP1); \
		JSONL_TOP4=$(REGRESSION_JSONL_DEC_TOP4); \
	fi; \
	STATUS=0; \
	if [ ! -f "$$ENCODER_PTE" ] || [ ! -f "$$VOCAB" ] || [ ! -f "$$LAYOUT" ]; then \
		echo "Error: missing encoder/vocab/layout. Cannot run regression" >&2; \
		exit 1; \
	fi; \
	if [ -f "$(JSONL_DATASET)" ]; then \
		echo "=== regression: infer_jsonl [$$DEC_LABEL, $(JSONL_LIMIT) samples] ==="; \
		TMP_JSONL=$$(mktemp); TMP_OUT=$$(mktemp); \
		head -n $(JSONL_LIMIT) "$(JSONL_DATASET)" > "$$TMP_JSONL"; \
		if ! build/infer_jsonl --encoder "$$ENCODER_PTE" $$DEC_ARGS --vocab "$$VOCAB" --layout "$$LAYOUT" \
			--jsonl "$$TMP_JSONL" \
			--beam-width $(CONSIST_BEAM_WIDTH) --top-k $(CONSIST_TOP_K) --expansion > "$$TMP_OUT" 2>&1; then \
			cat "$$TMP_OUT"; rm -f "$$TMP_OUT" "$$TMP_JSONL"; exit 1; \
		fi; \
		sed -n '/^Results:/,$$p' "$$TMP_OUT"; \
		TOP1=$$(awk '/^  top-1:/ { gsub("%","",$$NF); printf "%d\n", $$NF + 0 }' "$$TMP_OUT"); \
		TOP4=$$(awk '/^  top-4:/ { gsub("%","",$$NF); printf "%d\n", $$NF + 0 }' "$$TMP_OUT"); \
		rm -f "$$TMP_OUT" "$$TMP_JSONL"; \
		if [ -z "$$TOP1" ] || [ -z "$$TOP4" ]; then \
			echo "ERROR: could not parse infer_jsonl output"; STATUS=1; \
		elif [ "$$TOP1" -lt "$$JSONL_TOP1" ] || [ "$$TOP4" -lt "$$JSONL_TOP4" ]; then \
			echo "FAIL: jsonl top-1=$$TOP1% (>=$$JSONL_TOP1% required)  top-4=$$TOP4% (>=$$JSONL_TOP4% required)"; \
			STATUS=1; \
		else \
			echo "PASS: jsonl top-1=$$TOP1% (>=$$JSONL_TOP1%)  top-4=$$TOP4% (>=$$JSONL_TOP4%)"; \
		fi; \
	else \
		echo "Skipping infer_jsonl regression: $(JSONL_DATASET) not found"; \
	fi; \
	if [ -f "$(EVAL_DATASET_BIN)" ]; then \
		echo "=== regression: eval_accuracy [$$DEC_LABEL, $(EVAL_LIMIT) samples] ==="; \
		TMP_JSON=$$(mktemp); \
		if ! build/eval_accuracy --encoder "$$ENCODER_PTE" $$DEC_ARGS --vocab "$$VOCAB" --layout "$$LAYOUT" \
			--dataset "$(EVAL_DATASET_BIN)" \
			--beam-width $(CONSIST_BEAM_WIDTH) --top-k $(CONSIST_TOP_K) \
			--expansion --limit $(EVAL_LIMIT) --json > "$$TMP_JSON" 2>&1; then \
			cat "$$TMP_JSON"; rm -f "$$TMP_JSON"; exit 1; \
		fi; \
		cat "$$TMP_JSON"; \
		TOP1=$$(awk -F'[:,]' '/top1_accuracy/ { gsub(/[ "%]/,"",$$2); printf "%d\n", $$2 * 100 }' "$$TMP_JSON"); \
		TOP4=$$(awk -F'[:,]' '/top4_accuracy/ { gsub(/[ "%]/,"",$$2); printf "%d\n", $$2 * 100 }' "$$TMP_JSON"); \
		rm -f "$$TMP_JSON"; \
		if [ -z "$$TOP1" ] || [ -z "$$TOP4" ]; then \
			echo "ERROR: could not parse eval_accuracy output"; STATUS=1; \
		elif [ "$$TOP1" -lt "$(REGRESSION_BIN_ENC_TOP1)" ] || [ "$$TOP4" -lt "$(REGRESSION_BIN_ENC_TOP4)" ]; then \
			echo "FAIL: bin top-1=$$TOP1% (>=$(REGRESSION_BIN_ENC_TOP1)% required)  top-4=$$TOP4% (>=$(REGRESSION_BIN_ENC_TOP4)% required)"; \
			STATUS=1; \
		else \
			echo "PASS: bin top-1=$$TOP1% (>=$(REGRESSION_BIN_ENC_TOP1)%)  top-4=$$TOP4% (>=$(REGRESSION_BIN_ENC_TOP4)%)"; \
		fi; \
	else \
		echo "Skipping eval_accuracy regression: $(EVAL_DATASET_BIN) not found"; \
	fi; \
	exit $$STATUS

lint:
	cppcheck --enable=all --std=c++17 \
		--suppress=missingIncludeSystem \
		--suppress=pointerOutOfBoundsCond:include/swipe_decoder/vocab_hash.hpp \
		-I include/ src/ jni/ tests/

# ===========================================================================
# Android (multi-ABI: arm64-v8a, armeabi-v7a, x86_64, x86)
# ===========================================================================

LLVM_STRIP := $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip

# Common Android cmake flags
ANDROID_CMAKE_FLAGS = \
	-DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN) \
	-DANDROID_PLATFORM=android-27 \
	-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON

# Map ABI to libc++ sysroot subdirectory for push target
LIBCXX_DIR_arm64-v8a     := aarch64-linux-android
LIBCXX_DIR_armeabi-v7a   := arm-linux-androideabi
LIBCXX_DIR_x86_64        := x86_64-linux-android
LIBCXX_DIR_x86           := i686-linux-android

# Generate per-ABI targets:
#   $(ET_ROOT)/cmake-out-<abi>/libexecutorch.a - ET build
#   build-android-<abi>/libswipe_jni.so        - ml-inference build
#   android-<abi>                              - phony build + strip
define ANDROID_ABI_RULES

.PRECIOUS: $(ET_ROOT)/cmake-out-$(1)/libexecutorch.a

$(ET_ROOT)/cmake-out-$(1)/libexecutorch.a: $(ET_ROOT)/CMakeLists.txt | $(VENV_PYTHON) patch-et
	@test -f $(ET_ROOT)/CMakeLists.txt || { echo "Error: ExecuTorch submodule not initialized. Run: git submodule update --init --recursive" >&2; exit 1; }
	@test -n "$(NDK)" || { echo "Error: Android NDK not found. Set ANDROID_NDK_HOME." >&2; exit 1; }
	@test -f "$(TOOLCHAIN)" || { echo "Error: NDK toolchain not found at $(TOOLCHAIN)" >&2; exit 1; }
	@echo "=== Building ExecuTorch ($(1)) ==="
	$(VENV_CMAKE) -B $(ET_ROOT)/cmake-out-$(1) -S $(ET_ROOT) \
		-DCMAKE_BUILD_TYPE=Release \
		$(ANDROID_CMAKE_FLAGS) -DANDROID_ABI=$(1) \
		$(ET_CMAKE_FLAGS)
	$(VENV_CMAKE) --build $(ET_ROOT)/cmake-out-$(1) -j$(JOBS)

build-android-$(1)/libswipe_jni.so: $(ET_ROOT)/cmake-out-$(1)/libexecutorch.a CMakeLists.txt $(wildcard src/*.cpp include/swipe_decoder/*.hpp jni/*.cpp tests/*.cpp targets/*.cpp benchmark/*.cpp)
	@test -f "$(TOOLCHAIN)" || { echo "Error: NDK toolchain not found at $(TOOLCHAIN)" >&2; exit 1; }
	cmake -B build-android-$(1) \
		$(ANDROID_CMAKE_FLAGS) -DANDROID_ABI=$(1) \
		-DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=Release .
	cmake --build build-android-$(1) -j$(JOBS)

.PHONY: android-$(1)
android-$(1): build-android-$(1)/libswipe_jni.so
	@if [ -x "$(LLVM_STRIP)" ]; then \
		[ -f build-android-$(1)/libswipe_jni.so ] && $(LLVM_STRIP) --strip-all build-android-$(1)/libswipe_jni.so || true; \
		[ -f build-android-$(1)/benchmark_e2e ] && $(LLVM_STRIP) --strip-all build-android-$(1)/benchmark_e2e || true; \
		[ -f build-android-$(1)/eval_accuracy ] && $(LLVM_STRIP) --strip-all build-android-$(1)/eval_accuracy || true; \
		[ -f build-android-$(1)/infer_jsonl ] && $(LLVM_STRIP) --strip-all build-android-$(1)/infer_jsonl || true; \
	fi
	@echo "Build complete ($(1)): build-android-$(1)/"
	@ls -lh build-android-$(1)/libswipe_jni.so build-android-$(1)/benchmark_e2e build-android-$(1)/eval_accuracy build-android-$(1)/infer_jsonl 2>/dev/null || true

endef

$(foreach abi,$(ANDROID_ABIS),$(eval $(call ANDROID_ABI_RULES,$(abi))))

.PHONY: android push push-demo push-eval benchmark eval-jsonl-android eval-futo-android test-android verify-demo export-bench

android: $(foreach abi,$(ANDROID_ABIS),android-$(abi))

# Push uses ABI auto-detection from connected device, or override with ABI=arm64-v8a
ABI ?= $(shell adb shell getprop ro.product.cpu.abi 2>/dev/null || echo arm64-v8a)

push: android-$(ABI)
	adb shell "mkdir -p $(DEVICE_DIR)"
	adb push build-android-$(ABI)/benchmark_e2e $(DEVICE_DIR)/
	adb push models/$(ENCODER)/$(ENCODER_FILE) $(DEVICE_DIR)/
	adb push models/$(ENCODER)/metadata.json $(DEVICE_DIR)/ 2>/dev/null || true
	adb push models/$(VOCAB_FILE) $(DEVICE_DIR)/ 2>/dev/null || true
	adb push models/$(LAYOUT_FILE) $(DEVICE_DIR)/ 2>/dev/null || true
	@if [ -n "$(LM_MODEL)" ] && [ -f models/$(LM_MODEL)/$(LM_MODEL_FILE) ]; then \
		adb shell "mkdir -p $(DEVICE_DIR)/$(LM_MODEL)"; \
		adb push models/$(LM_MODEL)/$(LM_MODEL_FILE) $(DEVICE_DIR)/$(LM_MODEL)/; \
		adb push models/$(LM_MODEL)/$(LM_VOCAB_FILE) $(DEVICE_DIR)/$(LM_MODEL)/ 2>/dev/null || true; \
		adb push models/$(LM_MODEL)/metadata.json $(DEVICE_DIR)/$(LM_MODEL)/ 2>/dev/null || true; \
	fi
	@if [ -n "$(DECODER)" ] && [ -f models/$(DECODER)/$(DECODER_FILE) ]; then \
		adb shell "mkdir -p $(DEVICE_DIR)/$(DECODER)"; \
		adb push models/$(DECODER)/$(DECODER_FILE) $(DEVICE_DIR)/$(DECODER)/; \
		adb push models/$(DECODER)/metadata.json $(DEVICE_DIR)/$(DECODER)/ 2>/dev/null || true; \
	fi
	@LIBCXX="$(NDK)/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/$(LIBCXX_DIR_$(ABI))/libc++_shared.so"; \
		if [ -f "$$LIBCXX" ]; then adb push "$$LIBCXX" $(DEVICE_DIR)/; fi
	adb shell "chmod +x $(DEVICE_DIR)/benchmark_e2e"

# Push everything needed for the demo app (models, layouts, language vocabs)
DICTIONARIES_DIR ?= ../android-keyboard/dictionaries

push-demo: push
	adb shell "mkdir -p $(DEVICE_DIR)/$(ENCODER) $(DEVICE_DIR)/$(LM_MODEL) $(DEVICE_DIR)/layouts $(DEVICE_DIR)/vocabs"
	adb push models/$(ENCODER)/. $(DEVICE_DIR)/$(ENCODER)/
	adb push models/$(LM_MODEL)/. $(DEVICE_DIR)/$(LM_MODEL)/
	adb push models/layouts/. $(DEVICE_DIR)/layouts/
	adb push models/vocabs/. $(DEVICE_DIR)/vocabs/
	@if [ -n "$(DECODER)" ] && [ -d models/$(DECODER) ]; then \
		adb shell "mkdir -p $(DEVICE_DIR)/$(DECODER)"; \
		adb push models/$(DECODER)/. $(DEVICE_DIR)/$(DECODER)/; \
	fi
	@for lang in fr de es ru lv; do \
		gz="$(DICTIONARIES_DIR)/$${lang}_wordlist.combined.gz"; \
		if [ -f "$$gz" ]; then \
			echo "Decompressing $$gz -> $(DEVICE_DIR)/vocabs/$${lang}_wordlist.combined"; \
			gunzip -c "$$gz" | adb shell "cat > $(DEVICE_DIR)/vocabs/$${lang}_wordlist.combined"; \
		else \
			echo "Warning: $$gz not found, skipping"; \
		fi; \
	done

# End-to-end UI verification: drives the installed demo APK through every
# preset, layout swap, context flow, memory churn, and a batch of adversarial
# inputs. Catches loading/unloading regressions, memory leaks, and crashes.
# Requires a connected device.
verify-demo: push-demo
	cd android && ./gradlew :demo:installDebug
	python3 scripts/verify_demo.py $(VERIFY_DEMO_ARGS)

benchmark: push
	@ENCODER_BASENAME="$(notdir $(ENCODER_FILE))"; \
	VOCAB_BASENAME="$(notdir $(VOCAB_FILE))"; \
	LAYOUT_BASENAME="$(notdir $(LAYOUT_FILE))"; \
	ARGS="--encoder $$ENCODER_BASENAME --vocab $$VOCAB_BASENAME --layout $$LAYOUT_BASENAME --threads 1 --warmup 50 --runs 500 --affinity $(BENCH_AFFINITY)"; \
	if [ -n "$(DECODER)" ] && [ -f models/$(DECODER)/$(DECODER_FILE) ]; then \
		ARGS="$$ARGS --decoder $(DECODER)/$(notdir $(DECODER_FILE))"; \
	fi; \
	adb shell "cd $(DEVICE_DIR) && LD_LIBRARY_PATH=. ./benchmark_e2e $$ARGS"

# Run the on-device benchmark twice (encoder-only, encoder+decoder) and drop
# benchmark.json outputs into models/<codename>/ and the matching
# universal-swipe exports/ dir, where the model-card renderer picks them up.
UNIVERSAL_SWIPE_DIR ?= ../universal-swipe
# Benchmarks run on both big and little by default so the model card can show
# steady-state alongside peak-throughput (big).
# Override with CARD_AFFINITIES=little to skip the big run.
CARD_AFFINITIES ?= big,little
export-bench: push-demo
	python3 scripts/export_bench.py \
		--encoder $(ENCODER) \
		--decoder $(DECODER) \
		--vocab $(VOCAB_FILE) \
		--layout $(LAYOUT_FILE) \
		--affinities $(CARD_AFFINITIES) \
		--universal-swipe $(UNIVERSAL_SWIPE_DIR)

# Push infer_jsonl + eval_accuracy + eval datasets for on-device accuracy tests
# Uses the same fp16 .pte models already staged by `push`.
push-eval: push
	adb push build-android-$(ABI)/infer_jsonl $(DEVICE_DIR)/
	adb push build-android-$(ABI)/eval_accuracy $(DEVICE_DIR)/
	adb shell "mkdir -p $(DEVICE_DIR)/eval"
	@if [ -f eval/swipe2_test.jsonl ]; then \
		if [ "$(JSONL_LIMIT_DEVICE)" -gt 0 ] 2>/dev/null; then \
			TMP=$$(mktemp); head -n $(JSONL_LIMIT_DEVICE) eval/swipe2_test.jsonl > "$$TMP"; \
			adb push "$$TMP" $(DEVICE_DIR)/eval/swipe2_test.jsonl; rm -f "$$TMP"; \
		else \
			adb push eval/swipe2_test.jsonl $(DEVICE_DIR)/eval/; \
		fi; \
	fi
	@if [ -f eval/futo_test.bin ]; then adb push eval/futo_test.bin $(DEVICE_DIR)/eval/; fi
	adb shell "chmod +x $(DEVICE_DIR)/infer_jsonl $(DEVICE_DIR)/eval_accuracy"

# Run JSONL accuracy test on device against the fp16 pipeline.
# JSONL_LIMIT_DEVICE=0 pushes the full dataset; set e.g. JSONL_LIMIT_DEVICE=500 for a quick pass.
JSONL_LIMIT_DEVICE ?= 0

eval-jsonl-android: push-eval
	@ENCODER_BASENAME="$(notdir $(ENCODER_FILE))"; \
	VOCAB_BASENAME="$(notdir $(VOCAB_FILE))"; \
	LAYOUT_BASENAME="$(notdir $(LAYOUT_FILE))"; \
	ARGS="--encoder $$ENCODER_BASENAME --vocab $$VOCAB_BASENAME --layout $$LAYOUT_BASENAME --jsonl eval/swipe2_test.jsonl --threads 1 --beam-width $(CONSIST_BEAM_WIDTH) --top-k $(CONSIST_TOP_K)"; \
	if [ -n "$(DECODER)" ] && [ -f models/$(DECODER)/$(DECODER_FILE) ]; then \
		ARGS="$$ARGS --decoder $(DECODER)/$(notdir $(DECODER_FILE))"; \
	fi; \
	if [ -f models/$(LM_MODEL)/$(LM_MODEL_FILE) ]; then \
		ARGS="$$ARGS --lm-model $(LM_MODEL)/$(LM_MODEL_FILE) --lm-vocab $(LM_MODEL)/$(LM_VOCAB_FILE)"; \
	fi; \
	adb shell "cd $(DEVICE_DIR) && LD_LIBRARY_PATH=. ./infer_jsonl $$ARGS 2>&1"

# Run the synthetic futo_test.bin eval on device (same fp16 pipeline).
eval-futo-android: push-eval
	@ENCODER_BASENAME="$(notdir $(ENCODER_FILE))"; \
	VOCAB_BASENAME="$(notdir $(VOCAB_FILE))"; \
	LAYOUT_BASENAME="$(notdir $(LAYOUT_FILE))"; \
	ARGS="--encoder $$ENCODER_BASENAME --vocab $$VOCAB_BASENAME --layout $$LAYOUT_BASENAME --dataset eval/futo_test.bin --beam-width $(CONSIST_BEAM_WIDTH) --top-k $(CONSIST_TOP_K) --expansion"; \
	if [ -n "$(DECODER)" ] && [ -f models/$(DECODER)/$(DECODER_FILE) ]; then \
		ARGS="$$ARGS --decoder $(DECODER)/$(notdir $(DECODER_FILE))"; \
	fi; \
	adb shell "cd $(DEVICE_DIR) && LD_LIBRARY_PATH=. ./eval_accuracy $$ARGS"

test-android: push
	cd android && ./gradlew connectedAndroidTest

# ===========================================================================
# Clean
# ===========================================================================

.PHONY: clean clean-host clean-android clean-asan clean-et clean-venv clean-all

clean: clean-host clean-android clean-asan

clean-host:
	rm -rf build

clean-android:
	rm -rf $(foreach abi,$(ANDROID_ABIS),build-android-$(abi)) android/build android/demo/build android/.gradle android/.cxx

clean-asan:
	rm -rf build-asan

clean-et:
	rm -rf $(ET_HOST) $(foreach abi,$(ANDROID_ABIS),$(ET_ROOT)/cmake-out-$(abi))
	rm -f $(ET_ROOT)/third-party/flatcc/lib/*.a $(ET_ROOT)/third-party/flatcc/bin/flatcc

clean-venv:
	rm -rf $(VENV)

clean-all: clean clean-et clean-venv
