# Fresh-Pod inference test

`test_model.py` is a read-only test of the delivered 768/8 MiniMind-V checkpoint
with SHA256 `4e40a7e96bd1e846f9f2c0a5ed29eecf6e7701480f0eb4a33f24aa603590df58`.
It uses the frozen SigLIP encoder and tokenizer from the same training bundle.
The test is run on a newly created A40 Pod with the existing Global Volume.

Inputs are six repository evaluation illustrations: dog and balloons, rainbow
umbrella, blue bicycle, yellow sports car, superhero with red cape, red racecar.
These were visually inspected before running inference. They are qualitative
sanity cases, not a representative or independently verified benchmark.

Per image: greedy caption, sampled caption with explicit repetition controls,
a short focused English question, and a black-image caption control. Three
images also receive a Korean prompt: 27 generation cases total. The complete
prompts, decoding parameters, outputs, timings, EOS and token repetition
measurements are recorded in `generations.json` and `results.txt`.

`image_conditioning.json` records image/black-image differences in next-token
probabilities and a 6×6 token-mean NLL table for short, manually observed image
descriptions. These are diagnostic probes; candidate lengths differ, and they
are not claimed to be a general image-recognition accuracy score.

The test checks the final weight hash before and after inference, finite
weights, exact loading of all nonvision keys, and successful encoder loading.
Reports, input images, source, dependency versions and SHA256 manifest are
saved under `tests/20261020-runpod` on shared-volume and copied locally. The
new test Pod alone is terminated after local checksum verification.
