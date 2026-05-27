# FUTO Swipe library

This is our open-source C++ library for swipe typing, using the FUTO Swipe models. It depends on ExecuTorch for model inference.

With this library and the FUTO Swipe models, you can do fast swipe typing recognition on mobile devices with arbitrary layouts and arbitrary languages, achieving competitive accuracy with major keyboards.

## Prerequisites

- CMake 3.29+, C++17 compiler
- Python 3 + `python3-venv`
- Android NDK r27+ (for device builds)
- ExecuTorch v1.2.0 (vendored as git submodule in `third_party/executorch`)


## Quick start

```bash
# Clone our repository
git clone --recursive https://gitlab.futo.org/keyboard/swipe-library

# Build for desktop Linux
make host
```

If you forgot to do a recursive clone, do `git submodule update --init --recursive` to make sure executorch is cloned.

To build the Android library:

```bash
make android
cd android
gradle build
ls -l build/outputs/aar/
```

## Usage details

The main public API is the `SwipeEngine` class defined in [engine.hpp](include/swipe_decoder/engine.hpp). This class is not thread-safe so you should take care to avoid concurrent usage. This section will explain some concepts to help you use this class.

### Models

An encoder model file path is required for creating a `SwipeEngine`. There must be a metadata.json file next to the file, so you'll probably want each model to live in its own subdirectory, like so:
```
SwipeModelFiles
├───encoder_model/
│       model_fp32.pte
│       metadata.json
├───english_decoder_model/
│       model_fp32.pte
│       metadata.json
└───english_contextlm/
        model_fp32.pte
        metadata.json
        vocab.txt
```

Decoder model is optional and requires a specific layout and language. It will not work with arbitrary languages or layouts.

ContextLM is optional and requires a specific language. It also requires the corresponding vocab file that was used during its training.

### Layout

You'll need to specify the layout you're using. There are some built-in functions to load one from a (simplfiied) JSON file, or you can use `default_en_qwerty()`. A layout contains letters and their corresponding positions on the layout, which should be normalized from 0.0 to 1.0. You can check [layout.hpp](include/swipe_decoder/layout.hpp) for more details.

The library currently doesn't check that the decoder model actually corresponds to your layout. So if you use a decoder but specify letters in the wrong order, or the coordinates are off, you will get poor results.

Duplicate letters in a layout are currently not permitted.

### Dictionary

The library API targets bring-your-own-dictionary. There is also a dictionary implementation included for convenience that loads AOSP dictionary raw `.combined` dictionaries (not compiled `.dict` ones), through `load_trie_simple(path)` in [trie.hpp](include/swipe_decoder/trie.hpp).

If you want to bring your own dictionary structure, you will need to implement the ITrie VTable for your dictionary, see [itrie.h](include/swipe_decoder/itrie.h).

If you're not sure what a trie is, you should [read up on it to better understand](https://www.cs.ucf.edu/~dmarino/progcontests/cop4516/notes/TrieNotes.pdf).

We only target trie dictionaries, but you may also be able to wrap other kinds of dictionaries in the interface.

You can specify multiple dictionaries, and different weighting for each one.

### Inputs

Your input x and y coordinates should be normalized from 0.0 to 1.0 and correspond to the layout you provided. The timestamps should contain milliseconds since the start of the swipe.

You should test your implementation carefully to make sure that, for example, tapping on the letter Q actually passes a coordinate corresponding to the center of the Q key you specified. It's very easy to mess this up, and you will have a bad time if you do!

For basic use, call `SwipeEngine::recognize(x, y, t, num_points)` and you'll get back a vector containing the predicted words. If you loaded a language model and want to use it for improved accuracy, specify `context_words`.

The library also supports recognizing two-finger simultaneous swipe input through the `SwipeEngine::recognize_multi` method.

### setMode

In a keyboard it's common to change layouts, languages, dictionaries during usage. You can use the `setMode` method on `SwipeEngine` to change the layout, dictionaries, decoder model, or context LM model during usage.

### Scoring

Scoring of results depends on the exact models loaded, it's important that you have a metadata.json next to each model file that identifies the model for this.

The file [scoring.json](models/scoring.json) is built into the library and includes currently supported model combinations. These parameters are used in [beam search](src/beam_search.cpp).

## Usage (Kotlin)

There are JNI bindings in [swipe_jni.cpp](jni/swipe_jni.cpp), and a Kotlin interface [SwipeDecoder.kt](android/src/main/kotlin/org/futo/ml/inference/SwipeDecoder.kt) which you can use.


## Example use in C++


```cpp
#include "swipe_decoder/engine.hpp"
#include "swipe_decoder/trie.hpp"
#include "swipe_decoder/layout.hpp"

auto simple_dictionary = *swipe_decoder::load_trie_simple("english.combined");

swipe_decoder::SwipeEngine engine;
swipe_decoder::SwipeEngine::Config cfg;
cfg.encoder_path = "models/encoder/model.pte"; // must have metadata.json next to it
cfg.decoder_path = ""; // empty = encoder-only
cfg.dictionaries = { simple_dictionary.itrie() };
cfg.layout = swipe_decoder::default_en_qwerty();
// Optional context LM (must have metadata.json next to it)
// cfg.lm_model_path = "models/contextlm/model.pte";
// cfg.lm_vocab_path = "models/contextlm/vocab.txt";
engine.init(cfg);

auto results = engine.recognize(x, y, t, num_points);
```

Runtime swap via `setMode` (each `std::optional` field is three-state: unset = keep, `""` on decoder/LM paths = unload, value = set/load):

```cpp
swipe_decoder::KeyboardLayout layout;
swipe_decoder::load_layout_json("layouts/fr_azerty.json", layout);

// Layout + dictionary swap
swipe_decoder::SwipeEngine::Mode mode;
mode.layout = layout;
mode.dictionaries = { my_other_dictionary.itrie(), my_other_dictionary2.itrie() };
engine.setMode(mode);

// Load a paired decoder for the active layout
engine.setMode({ .decoder_path = "models/new_decoder/model.pte" });

// Unload the paired decoder
engine.setMode({ .decoder_path = "" });

// Load / unload the context LM
engine.setMode({
    .lm_model_path = "models/hungry_jellyfish/context_lm.pte",
    .lm_vocab_path = "models/hungry_jellyfish/vocab.txt",
});
engine.setMode({ .lm_model_path = "" });
```


### ExecuTorch patch

`third_party/executorch` is pinned to v1.2.0 but requires one in-tree patch to build with `EXECUTORCH_OPTIMIZE_SIZE=ON`, located at `patches/executorch-xnnpack-fexceptions.patch`. It's applied automatically by `make host` / `make android`.

Because the patch modifies files *inside* the submodule's working tree, `git status` on the outer repo will show `third_party/executorch` as dirty after any build. To inspect or revert:

```bash
make patch-et   # apply patch (becomes dirty)
make unpatch-et # revert (becomes clean again)
git -C third_party/executorch diff # review what's applied
```

### Make targets

There are a bunch of make targets documented in the [Makefile](Makefile). Here are the most important ones to know about:
* `make host` - Builds [tests](tests/) and [targets](targets/) for desktop Linux. It'll build ExecuTorch if it hasn't already been built.
* `make android` - Builds `libswipe_jni.so` for the 4 Android ABIs: x86, x86_64, armv7, and armv8.
* `make clean` - Removes build artifacts, but does not clean the ExecuTorch build
* `make clean-et` - Removes the ExecuTorch build artifacts