"""M8 task 1: a fixed, reasonably diverse calibration passage -- "a few
hundred calibration tokens" (PROJECT_SPEC.md M8 task 1) covering several
different topics, so the per-channel selection frequencies calibration
measures aren't an artifact of one narrow subject. Kept as its own module
(not inlined in bench_m8_calibration.py) so other M8/M9 scripts that need
the same calibration set can import it without duplicating the text.
"""

CALIBRATION_TEXT = """
The history of computing is a story of abstraction. Each generation of
engineers built new tools on top of the ones that came before, hiding
complexity so the next generation could reach further. From vacuum tubes
to transistors, from assembly language to high-level compilers, the
pattern repeats: what was once an expert's craft becomes a beginner's
starting point.

Photosynthesis converts sunlight, water, and carbon dioxide into glucose
and oxygen. Inside the chloroplasts of a plant cell, chlorophyll absorbs
light energy and drives a chain of chemical reactions. The oxygen released
as a byproduct is what most animal life on Earth depends on to breathe,
making this one of the most consequential chemical processes in the
history of the planet.

Financial markets move on expectations as much as on facts. A company can
report strong earnings and still see its stock price fall, if investors
had expected even stronger results. This gap between what actually
happened and what people predicted would happen is often a better
explanation for short-term price movements than the underlying numbers
themselves.

The Roman aqueducts were engineering marvels built almost two thousand
years ago, carrying fresh water across great distances using nothing but
gravity and a precisely calculated gradient. Some of these structures
remained in use for centuries, a testament to the durability of careful
design over quick, expedient construction.

Machine learning models learn patterns from data rather than being
explicitly programmed with rules. A model trained to recognize handwritten
digits never receives an instruction like "a seven has a horizontal line
and a diagonal stroke" -- instead, it adjusts millions of internal
parameters until its own predictions match a large set of labeled
examples closely enough to generalize to new, unseen digits.

Climate patterns are shaped by ocean currents that redistribute heat
around the globe. The Gulf Stream, for instance, carries warm water from
the tropics toward Western Europe, moderating winters there compared to
other regions at similar latitudes. Disruption to these currents can have
outsized effects on regional weather far from where the disruption
originates.
""".strip()
